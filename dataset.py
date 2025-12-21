"""
TensorFlow Dataset for DocCornerNetV3.

Loads images and labels from:
- images/ directory (positive samples with documents)
- images-negative/ directory (negative samples without documents)
- labels/ directory (YOLO OBB format .txt files)
- Split files (train.txt, val.txt, test.txt)

Supports:
- Negative images (no document)
- Outlier images with specific augmentation config
- Full augmentation pipeline (geometric + color)
- Weighted sampling for outliers

Output format:
- image: [H, W, 3] float32, normalized with ImageNet stats
- coords: [8] float32, normalized [0,1] (x0,y0,x1,y1,x2,y2,x3,y3)
- has_doc: [1] float32, 1=document present, 0=negative sample
"""

import hashlib
import math
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import multiprocessing as mp

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default augmentation config
DEFAULT_AUG_CONFIG = {
    "rotation_degrees": 5,
    "scale_range": (0.9, 1.0),
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.1,
    "blur_prob": 0.1,
    "blur_kernel": 3,
    "translate": 0.0,
    "perspective": (0.0, 0.03),
}


def load_split_file(split_path: str) -> List[str]:
    """Load image names from split file."""
    with open(split_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_yolo_label(label_path: str) -> Tuple[np.ndarray, bool]:
    """
    Load coordinates from YOLO OBB format label file.

    Format: class_id x0 y0 x1 y1 x2 y2 x3 y3

    Returns:
        coords: [8] array of normalized coordinates
        has_doc: True if document present
    """
    if not os.path.exists(label_path):
        return np.zeros(8, dtype=np.float32), False

    with open(label_path, 'r') as f:
        line = f.readline().strip()

    if not line:
        return np.zeros(8, dtype=np.float32), False

    parts = line.split()
    if len(parts) < 9:
        return np.zeros(8, dtype=np.float32), False

    coords = np.array([float(x) for x in parts[1:9]], dtype=np.float32)
    return coords, True


def load_outlier_list(outlier_path: str) -> set:
    """Load list of outlier image names."""
    if not os.path.exists(outlier_path):
        return set()
    with open(outlier_path, 'r') as f:
        return {line.strip() for line in f if line.strip()}


def rotate_coords(coords: np.ndarray, angle_deg: float, aspect_ratio: float = 1.0) -> np.ndarray:
    """
    Rotate normalized coordinates around image center.

    When PIL rotates an image by +angle (counter-clockwise), the content moves
    counter-clockwise. To keep coordinates aligned with the rotated content,
    we rotate coordinates by -angle (clockwise).

    Args:
        coords: [8] array of normalized coordinates (x0,y0,x1,y1,...)
        angle_deg: Rotation angle in degrees (same as passed to PIL rotate)
        aspect_ratio: Width/height ratio of the image

    Returns:
        Rotated coordinates [8]
    """
    # Negate angle: PIL rotates image CCW, so coords rotate CW
    angle_rad = math.radians(-angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rotated = []
    for i in range(0, 8, 2):
        x_norm = coords[i]
        y_norm = coords[i + 1]

        # Scale x by aspect ratio for correct rotation in non-square images
        x_px = x_norm * aspect_ratio
        y_px = y_norm

        # Center point
        cx = 0.5 * aspect_ratio
        cy = 0.5

        # Translate to origin
        x_px -= cx
        y_px -= cy

        # Rotate
        x_new_px = x_px * cos_a - y_px * sin_a
        y_new_px = x_px * sin_a + y_px * cos_a

        # Translate back
        x_new_px += cx
        y_new_px += cy

        # Scale back
        x_new = x_new_px / aspect_ratio
        y_new = y_new_px

        rotated.extend([x_new, y_new])

    return np.array(rotated, dtype=np.float32)


def apply_color_augmentation(image: Image.Image, cfg: dict) -> Image.Image:
    """Apply color augmentation using PIL."""
    # Brightness
    if cfg.get("brightness", 0) > 0:
        factor = random.uniform(1 - cfg["brightness"], 1 + cfg["brightness"])
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(factor)

    # Contrast
    if cfg.get("contrast", 0) > 0:
        factor = random.uniform(1 - cfg["contrast"], 1 + cfg["contrast"])
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(factor)

    # Saturation
    if cfg.get("saturation", 0) > 0:
        factor = random.uniform(1 - cfg["saturation"], 1 + cfg["saturation"])
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(factor)

    # Gaussian blur
    if random.random() < cfg.get("blur_prob", 0):
        kernel = cfg.get("blur_kernel", 3)
        image = image.filter(ImageFilter.GaussianBlur(radius=kernel / 2))

    return image


def apply_geometric_augmentation(
    image: Image.Image,
    coords: np.ndarray,
    cfg: dict,
    target_size: int = 224,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Apply geometric augmentation (rotation, scale, horizontal flip).

    IMPORTANT: First resize to square, then apply transforms.
    This is because coordinates are normalized to [0,1] assuming square image.
    """
    # First resize to square - coordinates are in normalized [0,1] square space
    image = image.resize((target_size, target_size), Image.BILINEAR)
    w, h = image.size  # Now w == h == target_size

    # Rotation (aspect_ratio = 1.0 since image is now square)
    rot_deg = cfg.get("rotation_degrees", 0)
    if rot_deg > 0:
        angle = random.uniform(-rot_deg, rot_deg)
        image = image.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(128, 128, 128))
        coords = rotate_coords(coords, angle, aspect_ratio=1.0)

    # Scale (zoom in/out)
    scale_range = cfg.get("scale_range", (1.0, 1.0))
    scale = random.uniform(*scale_range)
    coords = coords.copy()

    if scale < 1.0:
        # Zoom out: scale coordinates toward center
        for i in range(0, 8, 2):
            coords[i] = 0.5 + (coords[i] - 0.5) * scale
            coords[i + 1] = 0.5 + (coords[i + 1] - 0.5) * scale

        new_size = int(w * scale)
        scaled = image.resize((new_size, new_size), Image.BILINEAR)
        canvas = Image.new("RGB", (w, h), (128, 128, 128))
        offset = (w - new_size) // 2
        canvas.paste(scaled, (offset, offset))
        image = canvas

    elif scale > 1.0:
        # Zoom in: crop center
        for i in range(0, 8, 2):
            coords[i] = 0.5 + (coords[i] - 0.5) * scale
            coords[i + 1] = 0.5 + (coords[i + 1] - 0.5) * scale

        new_size = int(w * scale)
        scaled = image.resize((new_size, new_size), Image.BILINEAR)
        offset = (new_size - w) // 2
        image = scaled.crop((offset, offset, offset + w, offset + h))

    # Horizontal flip
    if cfg.get("horizontal_flip", False) and random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        # coords format: [x0,y0,x1,y1,x2,y2,x3,y3] = [TL,TR,BR,BL]
        # After flip: TL<->TR, BL<->BR, and x = 1 - x
        x0, y0, x1, y1, x2, y2, x3, y3 = coords
        coords = np.array([
            1.0 - x1, y1,  # new TL = old TR
            1.0 - x0, y0,  # new TR = old TL
            1.0 - x3, y3,  # new BR = old BL
            1.0 - x2, y2,  # new BL = old BR
        ], dtype=np.float32)

    return image, coords


def apply_full_augmentation(
    image: Image.Image,
    coords: np.ndarray,
    has_doc: bool,
    cfg: dict,
    target_size: int = 224,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Apply full augmentation pipeline.

    For positive samples: geometric + color augmentation
    For negative samples: resize + color augmentation only

    Returns image already resized to target_size.
    """
    if has_doc:
        # Geometric + color for positive samples (includes resize)
        image, coords = apply_geometric_augmentation(image, coords, cfg, target_size)
        image = apply_color_augmentation(image, cfg)
    else:
        # Resize + color only for negative samples (no coords to transform)
        image = image.resize((target_size, target_size), Image.BILINEAR)
        image = apply_color_augmentation(image, cfg)

    return image, coords


# =============================================================================
# TensorFlow GPU Augmentations (fast, runs on GPU)
# =============================================================================

def tf_augment_batch(images, coords, has_doc, img_size=224, is_outlier=None, image_norm: str = "imagenet"):
    """
    Apply augmentation to a batch using TensorFlow ops (runs on GPU).

    This is MUCH faster than PIL augmentations because it runs on GPU.
    Uses fully vectorized operations for maximum speed.

    Args:
        images: [B, H, W, 3] float32 tensor (already normalized)
        coords: [B, 8] float32 tensor
        has_doc: [B] float32 tensor
        img_size: Image size
        is_outlier: [B] float32 tensor (1.0 for outlier, 0.0 for normal) or None

    Returns:
        augmented_images: [B, H, W, 3] float32
        augmented_coords: [B, 8] float32
    """
    batch_size = tf.shape(images)[0]

    norm_mode = (image_norm or "imagenet").lower().strip()
    if norm_mode in {"zero_one", "0_1", "01"}:
        clip_min, clip_max = 0.0, 1.0
        brightness_scale = 1.0
    elif norm_mode in {"raw255", "0_255", "0255"}:
        clip_min, clip_max = 0.0, 255.0
        brightness_scale = 255.0
    else:
        # "imagenet" (or unknown): assume roughly standardized inputs
        clip_min, clip_max = -3.0, 3.0
        brightness_scale = 1.0

    # Define augmentation strengths (normal vs outlier)
    # Normal: brightness=0.2, contrast=0.8-1.2, saturation=0.85-1.15
    # Outlier: brightness=0.3, contrast=0.7-1.3, saturation=0.8-1.2
    if is_outlier is not None:
        is_outlier = tf.reshape(tf.cast(is_outlier, tf.float32), [batch_size, 1, 1, 1])
        # Interpolate augmentation strength based on outlier flag
        brightness_range = 0.2 + 0.1 * tf.squeeze(is_outlier, axis=[1, 2, 3])  # 0.2 or 0.3
        contrast_delta = 0.2 + 0.1 * tf.squeeze(is_outlier, axis=[1, 2, 3])    # 0.2 or 0.3
        sat_delta = 0.15 + 0.05 * tf.squeeze(is_outlier, axis=[1, 2, 3])       # 0.15 or 0.2
    else:
        brightness_range = tf.fill([batch_size], 0.2)
        contrast_delta = tf.fill([batch_size], 0.2)
        sat_delta = tf.fill([batch_size], 0.15)

    # Random color augmentations (vectorized, per-sample random values)
    # Brightness (per-sample strength)
    brightness_delta = tf.random.uniform([batch_size], -1.0, 1.0) * brightness_range * brightness_scale
    brightness_delta = tf.reshape(brightness_delta, [batch_size, 1, 1, 1])
    images = images + brightness_delta

    # Contrast (per-sample strength)
    contrast_min = 1.0 - contrast_delta
    contrast_max = 1.0 + contrast_delta
    contrast_factor = tf.random.uniform([batch_size]) * (contrast_max - contrast_min) + contrast_min
    contrast_factor = tf.reshape(contrast_factor, [batch_size, 1, 1, 1])
    mean = tf.reduce_mean(images, axis=[1, 2, 3], keepdims=True)
    images = (images - mean) * contrast_factor + mean

    # Saturation (per-sample strength)
    sat_min = 1.0 - sat_delta
    sat_max = 1.0 + sat_delta
    sat_factor = tf.random.uniform([batch_size]) * (sat_max - sat_min) + sat_min
    sat_factor = tf.reshape(sat_factor, [batch_size, 1, 1, 1])
    gray = tf.reduce_mean(images, axis=-1, keepdims=True)
    gray = tf.tile(gray, [1, 1, 1, 3])
    images = gray + sat_factor * (images - gray)

    # Random horizontal flip (vectorized)
    flip_mask = tf.random.uniform([batch_size]) > 0.5
    flip_mask_img = tf.reshape(flip_mask, [batch_size, 1, 1, 1])

    # Flip images
    images_flipped = tf.reverse(images, axis=[2])  # Flip width dimension
    images = tf.where(flip_mask_img, images_flipped, images)

    # Flip coordinates (vectorized)
    # coords format: [x0,y0,x1,y1,x2,y2,x3,y3] = [TL,TR,BR,BL]
    # After horizontal flip: TL<->TR, BL<->BR, and x = 1 - x
    flip_mask_coord = tf.cast(tf.reshape(flip_mask, [batch_size, 1]), tf.float32)
    has_doc_mask = tf.cast(tf.reshape(has_doc > 0.5, [batch_size, 1]), tf.float32)
    should_flip_coords = flip_mask_coord * has_doc_mask  # [B, 1]

    # Original coords: [x0,y0,x1,y1,x2,y2,x3,y3]
    # Flipped coords:  [1-x1,y1,1-x0,y0,1-x3,y3,1-x2,y2]
    x0, y0, x1, y1, x2, y2, x3, y3 = tf.unstack(coords, axis=1)

    coords_flipped = tf.stack([
        1.0 - x1, y1,  # new TL = old TR with flipped x
        1.0 - x0, y0,  # new TR = old TL with flipped x
        1.0 - x3, y3,  # new BR = old BL with flipped x
        1.0 - x2, y2,  # new BL = old BR with flipped x
    ], axis=1)

    coords = coords * (1.0 - should_flip_coords) + coords_flipped * should_flip_coords

    # Clip values
    images = tf.clip_by_value(images, clip_min, clip_max)
    coords = tf.clip_by_value(coords, 0.0, 1.0)

    return images, coords


def tf_augment_color_only(images):
    """
    Apply color-only augmentation (no geometric transforms).
    Safe to use with any images.

    Args:
        images: [B, H, W, 3] float32 tensor (normalized)

    Returns:
        augmented_images: [B, H, W, 3] float32
    """
    images = tf.image.random_brightness(images, max_delta=0.15)
    images = tf.image.random_contrast(images, lower=0.85, upper=1.15)
    images = tf.image.random_saturation(images, lower=0.85, upper=1.15)
    images = tf.clip_by_value(images, -3.0, 3.0)
    return images


# =============================================================================
# Image Caching System
# =============================================================================

def _load_and_resize_image(args: tuple) -> Tuple[str, Optional[np.ndarray]]:
    """Load and resize a single image (for parallel processing)."""
    image_name, image_dir, negative_dir, img_size = args

    # Determine path
    if image_name.startswith("negative_") and negative_dir.exists():
        image_path = negative_dir / image_name
    else:
        image_path = image_dir / image_name

    # Try common extensions if not found
    if not image_path.exists():
        stem = Path(image_name).stem
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = image_path.parent / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break

    if not image_path.exists():
        return image_name, None

    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((img_size, img_size), Image.BILINEAR)
        return image_name, np.array(img, dtype=np.uint8)
    except Exception as e:
        print(f"Warning: Failed to load {image_name}: {e}")
        return image_name, None


def preload_images_to_cache(
    image_list: List[str],
    image_dir: Path,
    negative_dir: Path,
    img_size: int = 224,
    cache_dir: Optional[str] = None,
    force_cache: bool = False,
    num_workers: int = 8,
) -> Dict[str, np.ndarray]:
    """
    Pre-load images into memory cache with optional disk persistence.

    Args:
        image_list: List of image names to load
        image_dir: Directory for positive images
        negative_dir: Directory for negative images
        img_size: Target image size
        cache_dir: Directory for disk cache (None = memory only)
        force_cache: Force regeneration of disk cache
        num_workers: Number of parallel workers

    Returns:
        Dictionary mapping image_name -> numpy array (uint8, RGB)
    """
    # Generate cache filename based on image list hash
    image_list_hash = hashlib.md5(",".join(sorted(image_list)).encode()).hexdigest()[:8]
    cache_filename = f"image_cache_{img_size}px_{image_list_hash}.pkl"

    # Try to load from disk cache
    if cache_dir and not force_cache:
        cache_path = Path(cache_dir) / cache_filename
        if cache_path.exists():
            print(f"Loading image cache from disk: {cache_path}")
            try:
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)
                if len(cache) == len(image_list):
                    print(f"  Loaded {len(cache)} images from cache")
                    return cache
                else:
                    print(f"  Cache size mismatch ({len(cache)} vs {len(image_list)}), regenerating...")
            except Exception as e:
                print(f"  Failed to load cache: {e}, regenerating...")

    # Load images in parallel
    print(f"Pre-loading {len(image_list)} images into cache...")
    cache = {}

    # Prepare arguments for parallel loading
    args_list = [
        (name, image_dir, negative_dir, img_size)
        for name in image_list
    ]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(_load_and_resize_image, args_list),
            total=len(args_list),
            desc="Caching images",
            unit="img"
        ))

    # Build cache dict
    for name, img_array in results:
        if img_array is not None:
            cache[name] = img_array

    print(f"  Cached {len(cache)}/{len(image_list)} images")

    # Save to disk if cache_dir specified
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = cache_path / cache_filename
        print(f"Saving cache to disk: {cache_file}")
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(cache, f)
            # Print cache file size
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            print(f"  Cache size: {size_mb:.1f} MB")
        except Exception as e:
            print(f"  Warning: Failed to save cache: {e}")

    return cache


def preload_labels_to_cache(
    image_list: List[str],
    label_dir: Path,
    cache_dir: Optional[str],
    split: str,
    force_cache: bool = False,
    num_workers: Optional[int] = None,
) -> Dict[str, Tuple[np.ndarray, bool]]:
    """
    Pre-load YOLO label files into memory with optional disk cache.

    Used by fast_mode to avoid repeatedly opening tens of thousands of small label
    files (can be very slow on network filesystems).
    """
    if num_workers is None:
        cpu_count = os.cpu_count() or 8
        # Labels are I/O-bound, so a bit of oversubscription is OK.
        num_workers = min(32, max(8, cpu_count * 2))

    # Cache filename based on split + image list hash
    image_list_hash = hashlib.md5(",".join(sorted(image_list)).encode()).hexdigest()[:8]
    cache_filename = f"labels_cache_{split}_{image_list_hash}.pkl"

    cache_path: Optional[Path] = None
    if cache_dir:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir_path / cache_filename

    # Try to load from disk
    if cache_path and cache_path.exists() and not force_cache:
        print(f"Loading label cache from disk: {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            if isinstance(cache, dict) and len(cache) == len(image_list):
                print(f"  Loaded {len(cache)} labels from cache")
                return cache
            print(f"  Cache size mismatch ({len(cache)} vs {len(image_list)}), regenerating...")
        except Exception as e:
            print(f"  Failed to load label cache: {e}, regenerating...")

    # Load labels in parallel
    print(f"Pre-loading {len(image_list)} labels into cache...")

    def load_one(name: str):
        if name.startswith("negative_"):
            return name, (np.zeros(8, dtype=np.float32), False)
        label_path = label_dir / f"{Path(name).stem}.txt"
        return name, load_yolo_label(str(label_path))

    labels_dict: Dict[str, Tuple[np.ndarray, bool]] = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for name, label in tqdm(
            executor.map(load_one, image_list),
            total=len(image_list),
            desc=f"Loading labels ({split})",
            unit="img",
        ):
            labels_dict[name] = label

    # Save to disk
    if cache_path:
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        print(f"Saving label cache to disk: {cache_path}")
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(labels_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            print(f"  Label cache size: {size_mb:.1f} MB")
        except Exception as e:
            print(f"  Warning: Failed to save label cache: {e}")
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    return labels_dict


class DocCornerDataset:
    """
    Dataset class for document corner detection with proper augmentation.

    Supports:
    - Positive samples (with document corners)
    - Negative samples (no document)
    - Outlier samples with specific augmentation
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: int = 224,
        augment: bool = False,
        augment_config: Optional[dict] = None,
        augment_config_outlier: Optional[dict] = None,
        outlier_list: Optional[str] = None,
        negative_dir: str = "images-negative",
        shared_cache: Optional[Dict[str, np.ndarray]] = None,
        image_norm: str = "imagenet",
    ):
        """
        Args:
            data_root: Root directory containing images/, labels/, split files
            split: "train", "val", or "test"
            img_size: Target image size
            augment: Whether to apply augmentation
            augment_config: Augmentation config for normal samples
            augment_config_outlier: Augmentation config for outlier samples
            outlier_list: Path to file listing outlier image names
            negative_dir: Directory name for negative images
            shared_cache: Pre-loaded image cache (dict of name -> numpy array)
        """
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.augment = augment
        self.negative_dir = negative_dir
        self.shared_cache = shared_cache
        self.image_norm = str(image_norm)

        # Setup directories
        self.image_dir = self.data_root / "images"
        self.label_dir = self.data_root / "labels"
        self.negative_image_dir = self.data_root / negative_dir

        # Augmentation configs
        self.aug_config = DEFAULT_AUG_CONFIG.copy()
        if augment_config:
            self.aug_config.update(augment_config)

        self.aug_config_outlier = augment_config_outlier or self.aug_config

        # Load outliers
        self.outlier_names = set()
        if outlier_list:
            self.outlier_names = load_outlier_list(outlier_list)
            if self.outlier_names:
                print(f"  Loaded {len(self.outlier_names)} outlier names")

        # Load split file
        split_file = self._find_split_file(split)
        self.image_list = load_split_file(str(split_file))

        # Separate positive and negative samples
        self.positive_samples = []
        self.negative_samples = []

        for name in self.image_list:
            if name.startswith("negative_"):
                self.negative_samples.append(name)
            else:
                self.positive_samples.append(name)

        print(f"DocCornerDataset: Loaded {len(self.image_list)} images from {split_file.name}")
        print(f"  Positive: {len(self.positive_samples)}, Negative: {len(self.negative_samples)}")
        if self.shared_cache:
            print(f"  Using shared cache ({len(self.shared_cache)} images)")

        # Count outliers in positive samples
        outlier_count = sum(1 for name in self.positive_samples if name in self.outlier_names)
        if outlier_count > 0:
            print(f"  Outliers: {outlier_count}")

    def _find_split_file(self, split: str) -> Path:
        """Find the appropriate split file."""
        candidates = [
            self.data_root / f"{split}_with_negative_v2.txt",
            self.data_root / f"{split}_with_negative.txt",
            self.data_root / f"{split}.txt",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"No split file found for {split} in {self.data_root}")

    def __len__(self) -> int:
        return len(self.image_list)

    def _load_image(self, image_name: str, from_cache: bool = True) -> Image.Image:
        """
        Load image from cache or disk.

        Args:
            image_name: Image filename
            from_cache: Whether to use shared cache if available

        Returns:
            PIL Image (already resized if from cache)
        """
        # Try to load from cache first
        if from_cache and self.shared_cache is not None and image_name in self.shared_cache:
            np_img = self.shared_cache[image_name]
            return Image.fromarray(np_img)

        # Load from disk
        if image_name.startswith("negative_") and self.negative_image_dir.exists():
            image_path = self.negative_image_dir / image_name
        else:
            image_path = self.image_dir / image_name

        # Try common extensions if not found
        if not image_path.exists():
            stem = Path(image_name).stem
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                candidate = image_path.parent / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

        if image_path.exists():
            return Image.open(image_path).convert("RGB")
        else:
            # Return blank gray image if not found
            return Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))

    def _load_label(self, image_name: str) -> Tuple[np.ndarray, bool]:
        """Load label for image."""
        if image_name.startswith("negative_"):
            return np.zeros(8, dtype=np.float32), False

        label_path = self.label_dir / f"{Path(image_name).stem}.txt"
        return load_yolo_label(str(label_path))

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample."""
        image_name = self.image_list[idx]

        # Check if image comes from cache (already resized)
        from_cache = (self.shared_cache is not None and image_name in self.shared_cache)

        # Load image
        image = self._load_image(image_name)

        # Load label
        coords, has_doc = self._load_label(image_name)

        # Apply augmentation (includes resize to img_size)
        if self.augment:
            is_outlier = image_name in self.outlier_names
            cfg = self.aug_config_outlier if is_outlier else self.aug_config
            image, coords = apply_full_augmentation(image, coords, has_doc, cfg, self.img_size)
        elif not from_cache:
            # Only resize if not from cache (cache images are already resized)
            image = image.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Convert to numpy
        image_np = np.array(image, dtype=np.float32)

        # Input normalization
        # - "imagenet": [0,255] -> [0,1] -> (x-mean)/std
        # - "zero_one": [0,255] -> [0,1]
        # - "raw255":   [0,255] (no scaling)
        norm_mode = self.image_norm.lower().strip()
        if norm_mode == "imagenet":
            image_np = image_np / 255.0
            mean = np.array(IMAGENET_MEAN, dtype=np.float32)
            std = np.array(IMAGENET_STD, dtype=np.float32)
            image_np = (image_np - mean) / std
        elif norm_mode in {"zero_one", "0_1", "01"}:
            image_np = image_np / 255.0
        elif norm_mode in {"raw255", "0_255", "0255"}:
            pass
        else:
            raise ValueError(
                f"Unsupported image_norm='{self.image_norm}'. Use: imagenet, zero_one, raw255."
            )

        # Clamp coords
        coords = np.clip(coords, 0.0, 1.0)

        return {
            "image": image_np,
            "coords": coords,
            "has_doc": np.float32(1.0 if has_doc else 0.0),
            "image_name": image_name,
        }

    def get_sample_for_visualization(self, idx: int) -> dict:
        """
        Get sample with both original and augmented versions for visualization.

        Returns dict with:
        - original_image: PIL Image (resized, not augmented)
        - augmented_image: PIL Image (resized, augmented)
        - original_coords: [8] coords before augmentation
        - augmented_coords: [8] coords after augmentation
        - has_doc: bool
        - is_outlier: bool
        - image_name: str
        """
        image_name = self.image_list[idx]

        # Load image
        image = self._load_image(image_name)
        original_image = image.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Load label
        coords, has_doc = self._load_label(image_name)
        original_coords = coords.copy()

        # Apply augmentation (includes resize to img_size)
        is_outlier = image_name in self.outlier_names
        cfg = self.aug_config_outlier if is_outlier else self.aug_config
        aug_image, aug_coords = apply_full_augmentation(
            image.copy(), coords.copy(), has_doc, cfg, self.img_size
        )
        # aug_image is already resized to img_size by apply_full_augmentation

        return {
            "original_image": original_image,
            "augmented_image": aug_image,
            "original_coords": original_coords,
            "augmented_coords": aug_coords,
            "has_doc": has_doc,
            "is_outlier": is_outlier,
            "image_name": image_name,
        }


def create_tf_dataset(
    dataset: DocCornerDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    drop_remainder: bool = True,
    num_parallel_calls: int = None,
) -> tf.data.Dataset:
    """
    Create TensorFlow dataset from DocCornerDataset with parallel processing.

    Args:
        dataset: DocCornerDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        drop_remainder: Whether to drop incomplete batches
        num_parallel_calls: Number of parallel calls for map (None = AUTOTUNE)

    Returns:
        tf.data.Dataset yielding (image, {"has_doc": ..., "coords": ...})
    """
    if num_parallel_calls is None:
        num_parallel_calls = tf.data.AUTOTUNE

    # Create dataset from indices
    indices = np.arange(len(dataset), dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices(indices)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(indices), reshuffle_each_iteration=True)

    # Use py_function to load samples in parallel
    def load_sample(idx):
        idx = int(idx.numpy())
        sample = dataset[idx]
        return (
            sample["image"].astype(np.float32),
            sample["has_doc"],
            sample["coords"].astype(np.float32),
        )

    def tf_load_sample(idx):
        image, has_doc, coords = tf.py_function(
            load_sample,
            [idx],
            [tf.float32, tf.float32, tf.float32]
        )
        image.set_shape([dataset.img_size, dataset.img_size, 3])
        has_doc.set_shape([])
        coords.set_shape([8])
        return image, {"has_doc": has_doc, "coords": coords}

    ds = ds.map(tf_load_sample, num_parallel_calls=num_parallel_calls)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def create_weighted_tf_dataset(
    dataset: DocCornerDataset,
    batch_size: int = 32,
    outlier_weight: float = 3.0,
    drop_remainder: bool = True,
    num_parallel_calls: int = None,
) -> tf.data.Dataset:
    """
    Create TensorFlow dataset with weighted sampling for outliers.

    Outliers are sampled more frequently based on outlier_weight.
    Uses parallel processing for faster data loading.
    """
    if num_parallel_calls is None:
        num_parallel_calls = tf.data.AUTOTUNE

    # Create weights for sampling
    weights = []
    for name in dataset.image_list:
        if name in dataset.outlier_names:
            weights.append(outlier_weight)
        else:
            weights.append(1.0)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Pre-sample indices for one epoch
    indices = list(range(len(dataset)))
    num_samples = len(dataset)
    sampled_indices = random.choices(indices, weights=weights, k=num_samples)

    ds = tf.data.Dataset.from_tensor_slices(sampled_indices)

    # Use py_function to load samples in parallel
    def load_sample(idx):
        idx = idx.numpy()
        sample = dataset[idx]
        return (
            sample["image"].astype(np.float32),
            sample["has_doc"],
            sample["coords"].astype(np.float32),
        )

    def tf_load_sample(idx):
        image, has_doc, coords = tf.py_function(
            load_sample,
            [idx],
            [tf.float32, tf.float32, tf.float32]
        )
        image.set_shape([dataset.img_size, dataset.img_size, 3])
        has_doc.set_shape([])
        coords.set_shape([8])
        return image, {"has_doc": has_doc, "coords": coords}

    ds = ds.map(tf_load_sample, num_parallel_calls=num_parallel_calls)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def create_fast_cached_dataset(
    image_list: List[str],
    labels_dict: Dict[str, Tuple[np.ndarray, bool]],
    shared_cache: Dict[str, np.ndarray],
    img_size: int = 224,
    batch_size: int = 32,
    shuffle: bool = True,
    drop_remainder: bool = True,
    outlier_set: Optional[set] = None,
    outlier_weight: float = 1.0,
    image_norm: str = "imagenet",
) -> tf.data.Dataset:
    """
    Create ultra-fast dataset from numpy cache.

    Stores images as uint8 (~3.5GB for 23k images) and normalizes on GPU.

    Args:
        image_list: List of image names
        labels_dict: Dict mapping image_name -> (coords, has_doc)
        shared_cache: Dict mapping image_name -> numpy array [H,W,3] uint8
        img_size: Image size (for validation)
        batch_size: Batch size
        shuffle: Whether to shuffle
        drop_remainder: Drop incomplete batches
        outlier_set: Set of outlier image names (for stronger augmentation)

    Returns:
        tf.data.Dataset yielding (image, {"has_doc": ..., "coords": ..., "is_outlier": ...})
        Images are float32, normalized with ImageNet stats.
    """
    n_samples = len(image_list)
    outlier_set = outlier_set or set()

    if shuffle and outlier_set and outlier_weight and outlier_weight > 1.0:
        weights = [outlier_weight if name in outlier_set else 1.0 for name in image_list]
        indices = list(range(n_samples))
        sampled_indices = random.choices(indices, weights=weights, k=n_samples)
        image_list = [image_list[i] for i in sampled_indices]

    # Store images as uint8 to save memory (~3.5GB instead of ~14GB)
    all_images = np.zeros((n_samples, img_size, img_size, 3), dtype=np.uint8)
    all_coords = np.zeros((n_samples, 8), dtype=np.float32)
    all_has_doc = np.zeros((n_samples,), dtype=np.float32)
    all_is_outlier = np.zeros((n_samples,), dtype=np.float32)

    outlier_count = 0
    print(f"Preparing fast dataset with {n_samples} samples...")
    for i, name in enumerate(tqdm(image_list, desc="Preparing tensors", unit="img")):
        # Get image from cache (keep as uint8)
        if name in shared_cache:
            all_images[i] = shared_cache[name]

        # Get label
        if name in labels_dict:
            coords, has_doc = labels_dict[name]
            all_coords[i] = coords
            all_has_doc[i] = 1.0 if has_doc else 0.0

        # Check if outlier
        if name in outlier_set:
            all_is_outlier[i] = 1.0
            outlier_count += 1

    if outlier_count > 0:
        print(f"  Outliers in dataset: {outlier_count}")

    # Create TF dataset from uint8 images
    ds = tf.data.Dataset.from_tensor_slices({
        "image": all_images,  # uint8, will normalize on GPU
        "coords": all_coords,
        "has_doc": all_has_doc,
        "is_outlier": all_is_outlier,
    })

    # Free the numpy arrays to save memory
    del all_images

    if shuffle:
        ds = ds.shuffle(buffer_size=min(n_samples, 10000), reshuffle_each_iteration=True)

    def normalize_and_format(sample):
        # Convert uint8 to float32 and normalize (fast on GPU)
        norm_mode = image_norm.lower().strip()
        image = tf.cast(sample["image"], tf.float32)
        if norm_mode == "imagenet":
            image = image / 255.0
            mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
            std = tf.constant(IMAGENET_STD, dtype=tf.float32)
            image = (image - mean) / std
        elif norm_mode in {"zero_one", "0_1", "01"}:
            image = image / 255.0
        elif norm_mode in {"raw255", "0_255", "0255"}:
            pass
        else:
            raise ValueError(
                f"Unsupported image_norm='{image_norm}'. Use: imagenet, zero_one, raw255."
            )
        return image, {
            "has_doc": sample["has_doc"],
            "coords": sample["coords"],
            "is_outlier": sample["is_outlier"],
        }

    ds = ds.map(normalize_and_format, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def create_dataset(
    data_root: str,
    split: str = "train",
    img_size: int = 224,
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
    augment_config: Optional[dict] = None,
    augment_config_outlier: Optional[dict] = None,
    outlier_list: Optional[str] = None,
    outlier_weight: float = 1.0,
    negative_dir: str = "images-negative",
    shared_cache: Optional[Dict[str, np.ndarray]] = None,
    fast_mode: bool = False,
    drop_remainder: bool = True,
    image_norm: str = "imagenet",
) -> tf.data.Dataset:
    """
    Create TensorFlow dataset for DocCornerNetV3.

    Args:
        data_root: Root directory containing images/, labels/, split files
        split: "train", "val", or "test"
        img_size: Target image size
        batch_size: Batch size
        shuffle: Whether to shuffle
        augment: Whether to apply augmentation (training only) - IGNORED in fast_mode
        augment_config: Augmentation config for normal samples
        augment_config_outlier: Augmentation config for outlier samples
        outlier_list: Path to outlier list file
        outlier_weight: Weight multiplier for outlier sampling
        negative_dir: Directory name for negative images
        shared_cache: Pre-loaded image cache (dict of name -> numpy array)
        fast_mode: If True, uses ultra-fast tensor loading (requires shared_cache)
                   Augmentation is done in TensorFlow on GPU instead of PIL.

    Returns:
        tf.data.Dataset yielding (image, {"has_doc": ..., "coords": ...})
    """
    data_root_path = Path(data_root)

    # Fast mode: load all data into tensors upfront
    if fast_mode and shared_cache is not None:
        # Find split file
        split_file = None
        for prefix in [f"{split}_with_negative_v2", f"{split}_with_negative", split]:
            candidate = data_root_path / f"{prefix}.txt"
            if candidate.exists():
                split_file = candidate
                break

        if split_file is None:
            raise FileNotFoundError(f"No split file found for {split}")

        image_list = load_split_file(str(split_file))
        label_dir = data_root_path / "labels"

        # Pre-load all labels (can be slow on network FS; cache to disk).
        labels_dict = preload_labels_to_cache(
            image_list=image_list,
            label_dir=label_dir,
            cache_dir=str(data_root_path / ".cache"),
            split=split,
        )

        # Load outlier set if provided
        outlier_set = None
        if outlier_list:
            outlier_set = load_outlier_list(outlier_list)
            if outlier_set:
                print(f"  Loaded {len(outlier_set)} outliers for fast_mode")

        return create_fast_cached_dataset(
            image_list=image_list,
            labels_dict=labels_dict,
            shared_cache=shared_cache,
            img_size=img_size,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_remainder=drop_remainder,
            outlier_set=outlier_set,
            outlier_weight=outlier_weight,
            image_norm=image_norm,
        )

    # Standard mode with PIL augmentations
    dataset = DocCornerDataset(
        data_root=data_root,
        split=split,
        img_size=img_size,
        augment=augment,
        augment_config=augment_config,
        augment_config_outlier=augment_config_outlier,
        outlier_list=outlier_list,
        negative_dir=negative_dir,
        shared_cache=shared_cache,
        image_norm=image_norm,
    )

    if outlier_list and outlier_weight > 1.0 and shuffle:
        return create_weighted_tf_dataset(
            dataset,
            batch_size=batch_size,
            outlier_weight=outlier_weight,
            drop_remainder=drop_remainder,
        )
    else:
        return create_tf_dataset(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_remainder=drop_remainder,
        )


def create_dataloaders(
    data_root: str,
    img_size: int = 224,
    batch_size: int = 32,
    augment_config: Optional[dict] = None,
    augment_config_outlier: Optional[dict] = None,
    outlier_list: Optional[str] = None,
    outlier_weight: float = 1.0,
    negative_dir: str = "images-negative",
    train_split: str = "train",
    val_split: str = "val",
    cache_images: bool = False,
    cache_dir: Optional[str] = None,
    force_cache: bool = False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create train and validation dataloaders with optional image caching.

    Args:
        data_root: Root directory
        img_size: Image size
        batch_size: Batch size
        augment_config: Augmentation config for training
        augment_config_outlier: Augmentation config for outliers
        outlier_list: Path to outlier list file
        outlier_weight: Weight multiplier for outlier sampling
        negative_dir: Negative images directory name
        train_split: Train split name
        val_split: Val split name
        cache_images: Whether to pre-load images into RAM
        cache_dir: Directory for persistent disk cache (None = memory only)
        force_cache: Force regeneration of disk cache

    Returns:
        (train_ds, val_ds) tuple
    """
    data_root_path = Path(data_root)
    image_dir = data_root_path / "images"
    negative_image_dir = data_root_path / negative_dir

    shared_cache = None

    if cache_images:
        # Load both train and val image lists for unified cache
        train_split_file = None
        val_split_file = None

        for prefix in [f"{train_split}_with_negative_v2", f"{train_split}_with_negative", train_split]:
            candidate = data_root_path / f"{prefix}.txt"
            if candidate.exists():
                train_split_file = candidate
                break

        for prefix in [f"{val_split}_with_negative_v2", f"{val_split}_with_negative", val_split]:
            candidate = data_root_path / f"{prefix}.txt"
            if candidate.exists():
                val_split_file = candidate
                break

        all_images = []
        if train_split_file:
            all_images.extend(load_split_file(str(train_split_file)))
        if val_split_file:
            all_images.extend(load_split_file(str(val_split_file)))

        # Remove duplicates while preserving order
        seen = set()
        unique_images = []
        for img in all_images:
            if img not in seen:
                seen.add(img)
                unique_images.append(img)

        if unique_images:
            shared_cache = preload_images_to_cache(
                image_list=unique_images,
                image_dir=image_dir,
                negative_dir=negative_image_dir,
                img_size=img_size,
                cache_dir=cache_dir,
                force_cache=force_cache,
            )

    train_ds = create_dataset(
        data_root=data_root,
        split=train_split,
        img_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        augment=True,
        augment_config=augment_config,
        augment_config_outlier=augment_config_outlier,
        outlier_list=outlier_list,
        outlier_weight=outlier_weight,
        negative_dir=negative_dir,
        shared_cache=shared_cache,
    )

    val_ds = create_dataset(
        data_root=data_root,
        split=val_split,
        img_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        negative_dir=negative_dir,
        shared_cache=shared_cache,
    )

    return train_ds, val_ds


if __name__ == "__main__":
    import sys

    # Test dataset
    data_root = Path(__file__).parent.parent.parent / "datasets/official/doc-scanner-dataset-labeled"

    if not data_root.exists():
        print(f"Dataset not found at {data_root}")
        sys.exit(1)

    print("Testing DocCornerNetV3 Dataset...")
    print(f"Data root: {data_root}")

    # Test DocCornerDataset directly
    print("\n--- Testing DocCornerDataset ---")
    dataset = DocCornerDataset(
        data_root=str(data_root),
        split="train",
        img_size=224,
        augment=True,
        augment_config={
            "rotation_degrees": 5,
            "scale_range": (0.9, 1.0),
            "brightness": 0.2,
            "contrast": 0.2,
        },
    )

    print(f"Dataset size: {len(dataset)}")

    # Test a few samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Coords: {sample['coords'][:4]}... (first 4)")
        print(f"  Has doc: {sample['has_doc']}")
        print(f"  Name: {sample['image_name']}")

    # Test TF dataset
    print("\n--- Testing TF Dataset ---")
    train_ds, val_ds = create_dataloaders(
        data_root=str(data_root),
        img_size=224,
        batch_size=4,
    )

    print("\nTrain dataset sample:")
    for images, targets in train_ds.take(1):
        print(f"  Images shape: {images.shape}")
        print(f"  Coords shape: {targets['coords'].shape}")
        print(f"  Has_doc shape: {targets['has_doc'].shape}")
        print(f"  Has_doc values: {targets['has_doc'].numpy()}")

    print("\nDataset test passed!")
