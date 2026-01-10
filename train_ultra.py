"""
Ultra-optimized training script - cross-platform (A100/MPS/CPU).

Key optimizations:
1. Threading-based image loading (works everywhere, no semaphore issues)
2. Pre-allocated numpy arrays (no memory spikes)
3. tf.data pipeline with cache + prefetch
4. Mixed precision on CUDA
5. XLA JIT compilation on CUDA
6. Efficient training loop with minimal overhead

Supports multiple data sources:
- Local directory with images/, labels/, split files (--data_root)
- HuggingFace Hub dataset (--hf_dataset mapo80/DocCornerDataset)
- Local Parquet files (--hf_dataset ./hf_dataset)

Usage:
    # From local directory
    python train_ultra.py \
        --data_root /path/to/dataset \
        --output_dir /path/to/checkpoints \
        --backbone mobilenetv2 \
        --img_size 256 \
        --batch_size 512 \
        --epochs 100

    # From HuggingFace Hub
    python train_ultra.py \
        --hf_dataset mapo80/DocCornerDataset \
        --output_dir /path/to/checkpoints \
        --backbone mobilenetv2 \
        --img_size 256 \
        --batch_size 512 \
        --epochs 100

    # From local Parquet files
    python train_ultra.py \
        --hf_dataset ./hf_dataset \
        --output_dir /path/to/checkpoints \
        --backbone mobilenetv2 \
        --img_size 256 \
        --batch_size 512 \
        --epochs 100

    # Download HuggingFace dataset to local Parquet (no training)
    python train_ultra.py \
        --hf_dataset mapo80/DocCornerDataset \
        --download_hf ./downloaded_dataset
"""

import argparse
import json
import os
import shutil
import sys
import gc
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from PIL import Image


# ============================================================================
# Hard Sample Mining (OHEM)
# ============================================================================
class HardSampleTracker:
    """
    Tracks hard samples identified during validation for Online Hard Example Mining.

    Hard samples are those with any corner error >= threshold (default 20px).
    The tracker maintains a set of sample indices and their error scores,
    updated after each validation epoch.
    """

    def __init__(self, threshold_px: float = 20.0):
        self.threshold_px = threshold_px
        self.hard_indices = set()  # Set of training sample indices
        self.hard_scores = {}  # {idx: max_corner_error}

    def update_from_validation(self, sample_indices: np.ndarray, max_corner_errors: np.ndarray):
        """
        Update hard sample list after validation epoch.

        Args:
            sample_indices: Array of sample indices (global training indices)
            max_corner_errors: Array of max corner error per sample (in pixels)
        """
        self.hard_indices.clear()
        self.hard_scores.clear()
        for idx, err in zip(sample_indices, max_corner_errors):
            idx_int = int(idx)
            if err >= self.threshold_px:
                self.hard_indices.add(idx_int)
                self.hard_scores[idx_int] = float(err)

    def is_hard(self, idx: int) -> bool:
        """Check if a sample index is in the hard sample set."""
        return int(idx) in self.hard_indices

    def get_weight(self, idx: int, base_weight: float = 1.0, hard_weight: float = 2.0) -> float:
        """Get sample weight: base_weight + hard_weight if hard, else base_weight."""
        if self.is_hard(idx):
            return base_weight + hard_weight
        return base_weight

    def get_stats(self) -> tuple:
        """Return (num_hard_samples, avg_error_of_hard_samples)."""
        if not self.hard_scores:
            return 0, 0.0
        return len(self.hard_indices), float(np.mean(list(self.hard_scores.values())))

    def save(self, path: Path):
        """Save hard sample list to file."""
        path = Path(path)
        if self.hard_indices:
            indices = np.array(list(self.hard_indices), dtype=np.int64)
            scores = np.array([self.hard_scores[i] for i in indices], dtype=np.float32)
            np.savez(path, indices=indices, scores=scores)

    def load(self, path: Path):
        """Load hard sample list from file."""
        path = Path(path)
        if path.exists():
            data = np.load(path)
            self.hard_indices = set(data['indices'].tolist())
            self.hard_scores = dict(zip(data['indices'].tolist(), data['scores'].tolist()))


# ============================================================================
# Platform detection and configuration
# ============================================================================
def _normalize_backbone_weights(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None
    return value


def _find_init_weights_path(value: str) -> Path:
    """
    Resolve a warm-start weights path.
      - If `value` is a directory, look for common weights filenames inside it.
      - If `value` is a file, return it as-is.
    """
    p = Path(value).expanduser()
    if p.is_dir():
        for candidate in [
            "best_model.weights.h5",
            "final_model.weights.h5",
            "latest_weights.h5",
            "best_iou_weights.h5",
            "final_weights.h5",
            # Student/distillation artifacts (if present)
            "best_student.weights.h5",
            "final_student.weights.h5",
        ]:
            cand = p / candidate
            if cand.exists():
                return cand
        raise FileNotFoundError(f"Cannot find a weights file in init_weights directory: {p}")
    if p.exists():
        return p
    raise FileNotFoundError(f"init_weights not found: {p}")


def setup_platform():
    """Configure platform for maximum performance."""
    gpus = tf.config.list_physical_devices('GPU')

    print("\n" + "=" * 80, flush=True)
    print("Platform Configuration", flush=True)
    print("=" * 80, flush=True)

    if gpus:
        # Check for NVIDIA
        try:
            from tensorflow.python.client import device_lib
            devices = device_lib.list_local_devices()
            is_nvidia = False
            for d in devices:
                if 'GPU' in d.device_type:
                    desc = d.physical_device_desc.lower()
                    if 'nvidia' in desc or 'cuda' in desc:
                        is_nvidia = True
                        print(f"  GPU: {d.physical_device_desc}", flush=True)
                        break

            if is_nvidia:
                # NVIDIA GPU optimizations
                os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
                os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Enable mixed precision
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy('mixed_float16')
                print(f"  Mixed precision: mixed_float16", flush=True)
                print(f"  XLA JIT: enabled", flush=True)
                print("=" * 80 + "\n", flush=True)
                return 'cuda'
        except Exception as e:
            print(f"  GPU detection warning: {e}")

    # Check for MPS (Apple Silicon)
    if sys.platform == 'darwin':
        print("  Using Metal Performance Shaders (MPS)", flush=True)
        print("  Mixed precision: float32", flush=True)
        print("=" * 80 + "\n", flush=True)
        return 'mps'

    # CPU fallback
    cpu_count = os.cpu_count() or 4
    try:
        tf.config.threading.set_intra_op_parallelism_threads(cpu_count)
        tf.config.threading.set_inter_op_parallelism_threads(cpu_count)
    except RuntimeError as exc:
        print(f"  Threading config skipped: {exc}", flush=True)
    print(f"  Using CPU with {cpu_count} threads", flush=True)
    print("=" * 80 + "\n", flush=True)
    return 'cpu'


# ============================================================================
# Fast threaded image loading (works everywhere)
# ============================================================================

def load_single_image(args):
    """Load single image - thread-safe."""
    name, data_root, img_size = args
    data_root = Path(data_root)

    image_dir = data_root / "images"
    negative_dir = data_root / "images-negative"
    label_dir = data_root / "labels"

    coords = np.zeros(8, dtype=np.float32)
    has_doc = 0.0

    # Determine image path and load label
    if name.startswith("negative_"):
        img_path = negative_dir / name
        has_doc = 0.0
    else:
        img_path = image_dir / name
        label_path = label_dir / f"{Path(name).stem}.txt"
        if label_path.exists():
            try:
                with open(label_path) as f:
                    line = f.readline().strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 9:
                            coords = np.array([float(x) for x in parts[1:9]], dtype=np.float32)
                            has_doc = 1.0
            except:
                pass

    if not img_path.exists():
        return None

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.resize((img_size, img_size), Image.BILINEAR)
            img_array = np.asarray(img, dtype=np.uint8).copy()
        return (img_array, coords, has_doc)
    except:
        return None


def load_dataset_fast(data_root, split, img_size, num_workers=64):
    """Load dataset using threading - fast and portable."""
    data_root = Path(data_root)

    # Find split file
    split_file = data_root / f"{split}.txt"
    if not split_file.exists():
        for suffix in ["_with_negative_v2", "_with_negative"]:
            candidate = data_root / f"{split}{suffix}.txt"
            if candidate.exists():
                split_file = candidate
                break

    if not split_file.exists():
        raise FileNotFoundError(f"No split file found for {split} in {data_root}")

    with open(split_file) as f:
        image_names = [l.strip() for l in f if l.strip()]

    n_images = len(image_names)
    print(f"Loading {split}: {n_images} images from {split_file.name}", flush=True)
    print(f"  Using {num_workers} threads...", flush=True)

    start_time = time.time()

    # Prepare arguments
    args_list = [(name, str(data_root), img_size) for name in image_names]

    # Load with progress
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(load_single_image, args_list),
                          total=n_images, desc=f"Loading {split}", unit="img"):
            if result is not None:
                results.append(result)

    load_time = time.time() - start_time
    n_valid = len(results)

    # Pre-allocate and fill arrays (avoids memory spike from np.stack)
    print(f"  Stacking {n_valid} results into arrays...", flush=True)
    stack_start = time.time()

    images = np.empty((n_valid, img_size, img_size, 3), dtype=np.uint8)
    coords = np.empty((n_valid, 8), dtype=np.float32)
    has_doc = np.empty(n_valid, dtype=np.float32)

    for i, (img, c, h) in enumerate(
        tqdm(results, total=n_valid, desc=f"Stacking {split}", unit="img")
    ):
        images[i] = img
        coords[i] = c
        has_doc[i] = h

    del results
    gc.collect()

    stack_time = time.time() - stack_start
    total_time = time.time() - start_time

    mem_gb = images.nbytes / 1e9
    print(f"  Loaded {n_valid}/{n_images} valid images ({mem_gb:.2f} GB)", flush=True)
    print(f"  Time: {load_time:.1f}s load + {stack_time:.1f}s stack = {total_time:.1f}s total", flush=True)
    print(f"  Speed: {n_valid / load_time:.0f} img/s", flush=True)

    return images, coords, has_doc


# ============================================================================
# HuggingFace dataset loading
# ============================================================================

def download_hf_dataset(hf_dataset: str, output_dir: str, splits: list = None, hf_token: str = None):
    """
    Download a HuggingFace dataset directly as Parquet files (no processing/caching).

    Uses huggingface_hub to download raw parquet files directly, avoiding the
    datasets library which creates a large cache.

    Args:
        hf_dataset: HuggingFace dataset name (e.g., 'mapo80/DocCornerDataset')
        output_dir: Directory to save the Parquet files
        splits: List of splits to download (default: ['train', 'validation', 'test'])
        hf_token: HuggingFace API token (optional, for private datasets)
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub library not installed. "
            "Install with: pip install huggingface_hub"
        )

    # Get token from argument, environment, or HF CLI login
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if splits is None:
        splits = ["train", "val", "test"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset from HuggingFace: {hf_dataset}", flush=True)
    print(f"Output directory: {output_path}", flush=True)

    api = HfApi()

    # List all files in the dataset repo
    try:
        files = api.list_repo_files(repo_id=hf_dataset, repo_type="dataset", token=hf_token)
    except Exception as e:
        raise RuntimeError(f"Could not list files in {hf_dataset}: {e}")

    # Filter parquet files and group by split
    parquet_files = [f for f in files if f.endswith(".parquet")]
    print(f"Found {len(parquet_files)} parquet files in repository", flush=True)

    # Map split names for output directories
    split_dir_map = {
        "validation": "val",
        "val": "val",
        "train": "train",
        "test": "test",
    }

    for split in splits:
        # Find parquet files for this split (check various patterns)
        # HuggingFace uses patterns like: default/train/0000.parquet, train/data-00000.parquet, etc.
        split_files = [
            f for f in parquet_files
            if f"/{split}/" in f or f.startswith(f"{split}/") or f"/{split}-" in f or f.startswith(f"{split}-") or f"/{split}/" in f.lower()
        ]

        if not split_files:
            print(f"\nNo parquet files found for split '{split}', skipping...", flush=True)
            continue

        print(f"\nDownloading {split} split ({len(split_files)} files)...", flush=True)

        # Create split directory
        split_dir = split_dir_map.get(split, split)
        split_path = output_path / split_dir
        split_path.mkdir(parents=True, exist_ok=True)

        # Download each file
        for file_path in tqdm(split_files, desc=f"Downloading {split}", unit="file"):
            # Download to local cache and get path
            local_file = hf_hub_download(
                repo_id=hf_dataset,
                filename=file_path,
                repo_type="dataset",
                token=hf_token,
                local_dir=str(output_path),
                local_dir_use_symlinks=False,
            )

        # Move files from nested structure to split directory if needed
        # huggingface_hub downloads to output_path/split/filename.parquet
        downloaded_split_path = output_path / split
        if downloaded_split_path.exists() and downloaded_split_path != split_path:
            # Move files from downloaded location to our standard location
            for pf in downloaded_split_path.glob("*.parquet"):
                dest = split_path / pf.name
                if not dest.exists():
                    shutil.move(str(pf), str(dest))
            # Remove empty directory
            if downloaded_split_path.exists() and not list(downloaded_split_path.iterdir()):
                downloaded_split_path.rmdir()

        n_files = len(list(split_path.glob("*.parquet")))
        print(f"  Saved {n_files} parquet files to {split_path}", flush=True)

    print(f"\nDataset downloaded successfully to {output_path}", flush=True)
    print(f"You can now use: --hf_dataset {output_path}", flush=True)


def load_dataset_from_parquet(parquet_dir: str, split: str, img_size: int, num_workers: int = 64):
    """
    Load dataset directly from local Parquet files (no HF caching).

    Args:
        parquet_dir: Path to local Parquet directory (e.g., './hf_dataset')
        split: Split name ('train', 'validation', 'test')
        img_size: Target image size
        num_workers: Number of workers for parallel processing

    Returns:
        images: np.ndarray [N, H, W, 3] uint8
        coords: np.ndarray [N, 8] float32
        has_doc: np.ndarray [N] float32
    """
    import pyarrow.parquet as pq
    import io

    parquet_path = Path(parquet_dir)

    # Map split names (HF uses 'validation', local may use 'val')
    split_dir_map = {
        "validation": "val",
        "val": "val",
        "train": "train",
        "test": "test",
    }
    split_dir = split_dir_map.get(split, split)

    # Try multiple possible locations for split data
    possible_paths = [
        parquet_path / split_dir,           # e.g., ./hf_dataset/train/
        parquet_path / split,               # e.g., ./hf_dataset/validation/
        parquet_path / "data" / split_dir,  # e.g., ./hf_dataset/data/train/
        parquet_path / "data" / split,      # e.g., ./hf_dataset/data/validation/
        parquet_path,                       # e.g., ./hf_dataset/ (parquet files in root)
    ]

    split_path = None
    for p in possible_paths:
        if p.exists():
            # Check if it has parquet files (either directly or we'll check later)
            if p.is_dir():
                split_path = p
                break

    if split_path is None:
        raise FileNotFoundError(
            f"Split directory not found. Tried:\n" +
            "\n".join(f"  - {p}" for p in possible_paths) +
            f"\n\nContents of {parquet_path}:\n" +
            "\n".join(f"  - {x.name}" for x in parquet_path.iterdir()) if parquet_path.exists() else "  (directory doesn't exist)"
        )

    print(f"Loading {split} from local Parquet: {split_path}", flush=True)
    start_time = time.time()

    # Find all parquet files
    parquet_files = sorted(split_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {split_path}")

    print(f"  Found {len(parquet_files)} parquet files", flush=True)

    # Read all parquet files and collect rows
    all_rows = []
    for pf in tqdm(parquet_files, desc="Reading parquet", unit="file"):
        table = pq.read_table(pf)
        for i in range(len(table)):
            row = {
                "image_bytes": table["image"][i].as_py()["bytes"],
                "is_negative": table["is_negative"][i].as_py(),
                "corner_tl_x": table["corner_tl_x"][i].as_py(),
                "corner_tl_y": table["corner_tl_y"][i].as_py(),
                "corner_tr_x": table["corner_tr_x"][i].as_py(),
                "corner_tr_y": table["corner_tr_y"][i].as_py(),
                "corner_br_x": table["corner_br_x"][i].as_py(),
                "corner_br_y": table["corner_br_y"][i].as_py(),
                "corner_bl_x": table["corner_bl_x"][i].as_py(),
                "corner_bl_y": table["corner_bl_y"][i].as_py(),
            }
            all_rows.append(row)

    n_samples = len(all_rows)
    print(f"  Found {n_samples} samples in {split} split", flush=True)

    # Pre-allocate arrays
    images = np.empty((n_samples, img_size, img_size, 3), dtype=np.uint8)
    coords = np.zeros((n_samples, 8), dtype=np.float32)
    has_doc = np.zeros(n_samples, dtype=np.float32)

    def process_row(args):
        """Process a single row from parquet."""
        idx, row = args

        # Decode image
        img = Image.open(io.BytesIO(row["image_bytes"]))
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize
        img = img.resize((img_size, img_size), Image.BILINEAR)
        img_array = np.asarray(img, dtype=np.uint8).copy()

        # Get coordinates
        is_negative = row.get("is_negative", False)
        if is_negative or row.get("corner_tl_x") is None:
            sample_coords = np.zeros(8, dtype=np.float32)
            sample_has_doc = 0.0
        else:
            sample_coords = np.array([
                row["corner_tl_x"], row["corner_tl_y"],
                row["corner_tr_x"], row["corner_tr_y"],
                row["corner_br_x"], row["corner_br_y"],
                row["corner_bl_x"], row["corner_bl_y"],
            ], dtype=np.float32)
            sample_has_doc = 1.0

        return idx, img_array, sample_coords, sample_has_doc

    # Process samples in parallel
    print(f"  Processing {n_samples} samples with {num_workers} workers...", flush=True)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for idx, img_array, sample_coords, sample_has_doc in tqdm(
            executor.map(process_row, enumerate(all_rows)),
            total=n_samples,
            desc=f"Loading {split}",
            unit="img",
        ):
            images[idx] = img_array
            coords[idx] = sample_coords
            has_doc[idx] = sample_has_doc

    # Free memory
    del all_rows
    gc.collect()

    load_time = time.time() - start_time
    n_positive = int(has_doc.sum())
    n_negative = n_samples - n_positive
    mem_gb = images.nbytes / 1e9

    print(f"  Loaded {n_samples} images ({mem_gb:.2f} GB) in {load_time:.1f}s", flush=True)
    print(f"  Positive: {n_positive}, Negative: {n_negative}", flush=True)
    print(f"  Speed: {n_samples / load_time:.0f} img/s", flush=True)

    return images, coords, has_doc


def load_hf_dataset_to_numpy(hf_dataset: str, split: str, img_size: int, num_workers: int = 64):
    """
    Load dataset from HuggingFace (Hub or local Parquet files).

    Args:
        hf_dataset: HuggingFace dataset name (e.g., 'mapo80/DocCornerDataset')
                    or path to local Parquet directory (e.g., './hf_dataset')
        split: Split name ('train', 'validation', 'test')
        img_size: Target image size
        num_workers: Number of workers for parallel processing

    Returns:
        images: np.ndarray [N, H, W, 3] uint8
        coords: np.ndarray [N, 8] float32
        has_doc: np.ndarray [N] float32
    """
    # Check if this looks like a local path (contains / or \, or starts with .)
    is_local_path = "/" in hf_dataset or "\\" in hf_dataset or hf_dataset.startswith(".")

    if is_local_path:
        hf_path = Path(hf_dataset)
        if hf_path.exists() and hf_path.is_dir():
            return load_dataset_from_parquet(str(hf_path), split, img_size, num_workers)
        else:
            raise FileNotFoundError(
                f"Local dataset directory not found: {hf_path.absolute()}\n"
                f"Please download the dataset first with:\n"
                f"  python train_ultra.py --hf_dataset mapo80/DocCornerDataset --download_hf {hf_dataset}"
            )

    # HuggingFace Hub - use datasets library
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library not installed. "
            "Install with: pip install datasets"
        )

    print(f"Loading {split} from HuggingFace Hub: {hf_dataset}", flush=True)
    start_time = time.time()

    hf_ds = load_dataset(hf_dataset, split=split)

    n_samples = len(hf_ds)
    print(f"  Found {n_samples} samples in {split} split", flush=True)

    # Pre-allocate arrays
    images = np.empty((n_samples, img_size, img_size, 3), dtype=np.uint8)
    coords = np.zeros((n_samples, 8), dtype=np.float32)
    has_doc = np.zeros(n_samples, dtype=np.float32)

    def process_sample(idx):
        """Process a single sample from HF dataset."""
        sample = hf_ds[idx]
        img = sample["image"]

        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize
        img = img.resize((img_size, img_size), Image.BILINEAR)
        img_array = np.asarray(img, dtype=np.uint8).copy()

        # Get coordinates
        is_negative = sample.get("is_negative", False)
        if is_negative or sample.get("corner_tl_x") is None:
            sample_coords = np.zeros(8, dtype=np.float32)
            sample_has_doc = 0.0
        else:
            sample_coords = np.array([
                sample["corner_tl_x"], sample["corner_tl_y"],
                sample["corner_tr_x"], sample["corner_tr_y"],
                sample["corner_br_x"], sample["corner_br_y"],
                sample["corner_bl_x"], sample["corner_bl_y"],
            ], dtype=np.float32)
            sample_has_doc = 1.0

        return idx, img_array, sample_coords, sample_has_doc

    # Process samples in parallel
    print(f"  Processing {n_samples} samples with {num_workers} workers...", flush=True)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for idx, img_array, sample_coords, sample_has_doc in tqdm(
            executor.map(process_sample, range(n_samples)),
            total=n_samples,
            desc=f"Loading {split}",
            unit="img",
        ):
            images[idx] = img_array
            coords[idx] = sample_coords
            has_doc[idx] = sample_has_doc

    load_time = time.time() - start_time
    n_positive = int(has_doc.sum())
    n_negative = n_samples - n_positive
    mem_gb = images.nbytes / 1e9

    print(f"  Loaded {n_samples} images ({mem_gb:.2f} GB) in {load_time:.1f}s", flush=True)
    print(f"  Positive: {n_positive}, Negative: {n_negative}", flush=True)
    print(f"  Speed: {n_samples / load_time:.0f} img/s", flush=True)

    return images, coords, has_doc


# ============================================================================
# tf.data based dataset
# ============================================================================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGENET_MEAN_TF = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
IMAGENET_STD_TF = tf.constant(IMAGENET_STD, dtype=tf.float32)


class FastDataset:
    """Optimized dataset - keeps data in numpy, normalizes in tf.data."""

    def __init__(self, images, coords, has_doc, batch_size, shuffle=True, drop_remainder=False, name="dataset", return_indices=False):
        self.n_samples = len(images)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder
        self.name = name
        self.chunk_size = max(self.batch_size * 4, 1024)
        self.return_indices = return_indices

        # Store raw uint8 images (3GB instead of 12GB float32)
        self.images = images
        self.coords = coords.astype(np.float32)
        self.has_doc = has_doc.astype(np.float32)
        self.indices = np.arange(self.n_samples, dtype=np.int64)

        if drop_remainder:
            self.n_batches = self.n_samples // batch_size
        else:
            self.n_batches = (self.n_samples + batch_size - 1) // batch_size

        print("  Creating tf.data pipeline...", flush=True)
        dataset = self._build_base_dataset()
        if self.shuffle:
            dataset = dataset.shuffle(self.n_samples, reshuffle_each_iteration=True)

        dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
        dataset = dataset.map(self._normalize_batch, num_parallel_calls=tf.data.AUTOTUNE)
        self.dataset = dataset.prefetch(tf.data.AUTOTUNE)

        print(
            f"  Dataset ready: {self.n_batches} batches of {batch_size} "
            f"(drop_remainder={self.drop_remainder}, return_indices={self.return_indices})",
            flush=True,
        )

    def _build_base_dataset(self):
        if self.n_samples == 0:
            with tf.device("/CPU:0"):
                if self.return_indices:
                    return tf.data.Dataset.from_tensor_slices(
                        (self.indices, self.images, self.coords, self.has_doc)
                    )
                return tf.data.Dataset.from_tensor_slices(
                    (self.images, self.coords, self.has_doc)
                )

        dataset = None
        for start in tqdm(
            range(0, self.n_samples, self.chunk_size),
            desc=f"Building {self.name} dataset",
            unit="chunk",
        ):
            end = min(start + self.chunk_size, self.n_samples)
            if self.return_indices:
                shard = (self.indices[start:end], self.images[start:end], self.coords[start:end], self.has_doc[start:end])
            else:
                shard = (self.images[start:end], self.coords[start:end], self.has_doc[start:end])
            with tf.device("/CPU:0"):
                shard_ds = tf.data.Dataset.from_tensor_slices(shard)
            dataset = shard_ds if dataset is None else dataset.concatenate(shard_ds)

        return dataset

    def _normalize_batch(self, *args):
        """Normalize a batch with TF ops."""
        if self.return_indices:
            indices, images, coords, has_doc = args
            images = tf.cast(images, tf.float32) / 255.0
            images = (images - IMAGENET_MEAN_TF) / IMAGENET_STD_TF
            return indices, images, coords, has_doc
        else:
            images, coords, has_doc = args
            images = tf.cast(images, tf.float32) / 255.0
            images = (images - IMAGENET_MEAN_TF) / IMAGENET_STD_TF
            return images, coords, has_doc

    def reshuffle(self):
        """No-op: shuffle handled by tf.data."""
        return

    def __len__(self):
        return self.n_batches


# ============================================================================
# Training logic
# ============================================================================

from model import create_model, create_inference_model
from losses import gaussian_1d_targets
from metrics import ValidationMetrics
from dataset import tf_augment_batch


class Trainer:
    """Efficient trainer with compiled functions and gradient accumulation."""

    def __init__(self, model, optimizer, img_size, sigma_px, tau,
                 w_simcc, w_coord, w_score, platform='cuda', augment=False,
                 accumulation_steps=1):
        self.model = model
        self.optimizer = optimizer
        self.platform = platform
        self.use_mixed_precision = platform == 'cuda'
        self.augment = augment
        self.img_size = img_size
        self.accumulation_steps = accumulation_steps

        # Pre-compute constants as tensors
        self.img_size_tf = tf.constant(img_size, dtype=tf.int32)
        self.img_size_float = tf.constant(float(img_size), dtype=tf.float32)
        self.sigma_px = tf.constant(sigma_px, dtype=tf.float32)
        self.tau = tf.constant(tau, dtype=tf.float32)
        self.w_simcc = tf.constant(w_simcc, dtype=tf.float32)
        self.w_coord = tf.constant(w_coord, dtype=tf.float32)
        self.w_score = tf.constant(w_score, dtype=tf.float32)

        # Gradient accumulation buffers (initialized lazily)
        self._gradient_accumulator = None
        self._accumulation_count = tf.Variable(0, dtype=tf.int32, trainable=False)

    def _init_gradient_accumulator(self):
        """Initialize gradient accumulator with zeros matching model variables."""
        if self._gradient_accumulator is None:
            self._gradient_accumulator = [
                tf.Variable(tf.zeros_like(var), trainable=False)
                for var in self.model.trainable_variables
            ]

    def reset_gradient_accumulator(self):
        """Reset accumulated gradients to zero."""
        if self._gradient_accumulator is not None:
            for acc in self._gradient_accumulator:
                acc.assign(tf.zeros_like(acc))
        self._accumulation_count.assign(0)

    @tf.function
    def augment_batch(self, images, coords, has_doc):
        """Apply augmentation to batch."""
        return tf_augment_batch(images, coords, has_doc, self.img_size, image_norm="imagenet")

    def _compute_loss(self, images, coords_gt, has_doc, training, sample_weights=None):
        """Compute total loss and its components.

        Args:
            images: Input images [B, H, W, 3]
            coords_gt: Ground truth coordinates [B, 8]
            has_doc: Document presence mask [B]
            training: Boolean for training mode
            sample_weights: Optional per-sample weights [B] for OHEM (default: None, uses uniform weights)
        """
        outputs = self.model(images, training=training)

        # Cast to float32 for stable loss computation (model outputs are float16 with mixed precision)
        simcc_x = tf.cast(outputs["simcc_x"], tf.float32)
        simcc_y = tf.cast(outputs["simcc_y"], tf.float32)
        score_logit = tf.cast(outputs["score_logit"], tf.float32)
        coords_pred = tf.cast(outputs["coords"], tf.float32)

        # SimCC loss
        gt_coords_4x2 = tf.reshape(coords_gt, [-1, 4, 2])
        gt_x = gt_coords_4x2[:, :, 0]
        gt_y = gt_coords_4x2[:, :, 1]

        target_x = gaussian_1d_targets(gt_x, self.img_size_tf, self.sigma_px)
        target_y = gaussian_1d_targets(gt_y, self.img_size_tf, self.sigma_px)

        log_pred_x = tf.nn.log_softmax(simcc_x / self.tau, axis=-1)
        log_pred_y = tf.nn.log_softmax(simcc_y / self.tau, axis=-1)

        ce_x = -tf.reduce_sum(target_x * log_pred_x, axis=-1)
        ce_y = -tf.reduce_sum(target_y * log_pred_y, axis=-1)
        ce = tf.reduce_mean(ce_x + ce_y, axis=-1)  # [B]

        # Coord loss per sample
        loss_per_coord = tf.abs(coords_pred - coords_gt)
        loss_per_sample = tf.reduce_mean(loss_per_coord, axis=-1)  # [B]

        # Apply sample weights if provided (OHEM)
        if sample_weights is not None:
            # Weight both SimCC and coord losses
            weighted_mask = has_doc * sample_weights  # [B]
            loss_simcc = tf.reduce_sum(ce * weighted_mask) / (tf.reduce_sum(weighted_mask) + 1e-9)
            loss_coord = tf.reduce_sum(loss_per_sample * weighted_mask) / (tf.reduce_sum(weighted_mask) + 1e-9)
        else:
            loss_simcc = tf.reduce_sum(ce * has_doc) / (tf.reduce_sum(has_doc) + 1e-9)
            loss_coord = tf.reduce_sum(loss_per_sample * has_doc) / (tf.reduce_sum(has_doc) + 1e-9)

        # Score loss (not weighted - document detection should remain balanced)
        loss_score = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=has_doc[:, None],
            logits=score_logit
        )
        loss_score = tf.reduce_mean(loss_score)

        total_loss = self.w_simcc * loss_simcc + self.w_coord * loss_coord + self.w_score * loss_score

        return total_loss, coords_pred, loss_simcc, loss_coord, loss_score, score_logit

    def _batch_metrics(self, coords_pred, coords_gt, has_doc):
        """Compute IoU and corner error for positives in a batch."""
        img_size = tf.cast(self.img_size, tf.float32)
        has_doc_1d = tf.cast(has_doc, tf.float32)
        if len(has_doc_1d.shape) == 2:
            has_doc_1d = tf.squeeze(has_doc_1d, axis=-1)

        mask_bool = tf.cast(has_doc_1d, tf.bool)
        pred_pos = tf.boolean_mask(coords_pred, mask_bool)
        gt_pos = tf.boolean_mask(coords_gt, mask_bool)
        n_pos = tf.shape(pred_pos)[0]

        def compute_metrics():
            diff = tf.abs(pred_pos - gt_pos) * img_size
            corner_err = tf.reduce_mean(diff)

            pred_xy = tf.reshape(pred_pos, [-1, 4, 2])
            gt_xy = tf.reshape(gt_pos, [-1, 4, 2])

            pred_min = tf.reduce_min(pred_xy, axis=1)
            pred_max = tf.reduce_max(pred_xy, axis=1)
            gt_min = tf.reduce_min(gt_xy, axis=1)
            gt_max = tf.reduce_max(gt_xy, axis=1)

            inter_min = tf.maximum(pred_min, gt_min)
            inter_max = tf.minimum(pred_max, gt_max)
            inter_wh = tf.maximum(inter_max - inter_min, 0.0)
            inter_area = inter_wh[:, 0] * inter_wh[:, 1]

            pred_wh = pred_max - pred_min
            gt_wh = gt_max - gt_min
            pred_area = pred_wh[:, 0] * pred_wh[:, 1]
            gt_area = gt_wh[:, 0] * gt_wh[:, 1]

            union_area = pred_area + gt_area - inter_area + 1e-9
            iou = inter_area / union_area
            mean_iou = tf.reduce_mean(iou)

            return mean_iou, corner_err

        def zero_metrics():
            return tf.constant(0.0), tf.constant(0.0)

        return tf.cond(n_pos > 0, compute_metrics, zero_metrics)

    @tf.function
    def train_step(self, images, coords_gt, has_doc):
        """Training step (no accumulation - applies gradients immediately)."""
        with tf.GradientTape() as tape:
            (
                total_loss,
                coords_pred,
                loss_simcc,
                loss_coord,
                loss_score,
                _,
            ) = self._compute_loss(images, coords_gt, has_doc, training=True)
            if self.use_mixed_precision:
                if hasattr(self.optimizer, "scale_loss"):
                    scaled_loss = self.optimizer.scale_loss(total_loss)
                else:
                    scaled_loss = total_loss
            else:
                scaled_loss = total_loss

        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        iou, corner_err = self._batch_metrics(coords_pred, coords_gt, has_doc)
        return total_loss, loss_simcc, loss_coord, loss_score, iou, corner_err

    @tf.function
    def train_step_accumulate(self, images, coords_gt, has_doc):
        """Training step with gradient accumulation - accumulates gradients without applying."""
        with tf.GradientTape() as tape:
            (
                total_loss,
                coords_pred,
                loss_simcc,
                loss_coord,
                loss_score,
                _,
            ) = self._compute_loss(images, coords_gt, has_doc, training=True)
            if self.use_mixed_precision:
                if hasattr(self.optimizer, "scale_loss"):
                    scaled_loss = self.optimizer.scale_loss(total_loss)
                else:
                    scaled_loss = total_loss
            else:
                scaled_loss = total_loss

        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)

        # Accumulate gradients (add to existing)
        for acc, grad in zip(self._gradient_accumulator, gradients):
            if grad is not None:
                acc.assign_add(grad)

        self._accumulation_count.assign_add(1)
        iou, corner_err = self._batch_metrics(coords_pred, coords_gt, has_doc)
        return total_loss, loss_simcc, loss_coord, loss_score, iou, corner_err

    @tf.function
    def apply_accumulated_gradients(self):
        """Apply accumulated gradients (averaged over accumulation steps)."""
        # Average the gradients
        accum_steps_float = tf.cast(self._accumulation_count, tf.float32)
        averaged_gradients = [
            acc / accum_steps_float for acc in self._gradient_accumulator
        ]
        # Apply to optimizer
        self.optimizer.apply_gradients(
            zip(averaged_gradients, self.model.trainable_variables)
        )
        # Reset accumulators
        for acc in self._gradient_accumulator:
            acc.assign(tf.zeros_like(acc))
        self._accumulation_count.assign(0)

    @tf.function
    def train_step_weighted(self, images, coords_gt, has_doc, sample_weights):
        """Training step with sample weights (OHEM) - no accumulation."""
        with tf.GradientTape() as tape:
            (
                total_loss,
                coords_pred,
                loss_simcc,
                loss_coord,
                loss_score,
                _,
            ) = self._compute_loss(images, coords_gt, has_doc, training=True, sample_weights=sample_weights)
            if self.use_mixed_precision:
                if hasattr(self.optimizer, "scale_loss"):
                    scaled_loss = self.optimizer.scale_loss(total_loss)
                else:
                    scaled_loss = total_loss
            else:
                scaled_loss = total_loss

        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        iou, corner_err = self._batch_metrics(coords_pred, coords_gt, has_doc)
        return total_loss, loss_simcc, loss_coord, loss_score, iou, corner_err

    @tf.function
    def train_step_accumulate_weighted(self, images, coords_gt, has_doc, sample_weights):
        """Training step with sample weights (OHEM) and gradient accumulation."""
        with tf.GradientTape() as tape:
            (
                total_loss,
                coords_pred,
                loss_simcc,
                loss_coord,
                loss_score,
                _,
            ) = self._compute_loss(images, coords_gt, has_doc, training=True, sample_weights=sample_weights)
            if self.use_mixed_precision:
                if hasattr(self.optimizer, "scale_loss"):
                    scaled_loss = self.optimizer.scale_loss(total_loss)
                else:
                    scaled_loss = total_loss
            else:
                scaled_loss = total_loss

        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)

        # Accumulate gradients (add to existing)
        for acc, grad in zip(self._gradient_accumulator, gradients):
            if grad is not None:
                acc.assign_add(grad)

        self._accumulation_count.assign_add(1)
        iou, corner_err = self._batch_metrics(coords_pred, coords_gt, has_doc)
        return total_loss, loss_simcc, loss_coord, loss_score, iou, corner_err

    @tf.function
    def val_step(self, images, coords_gt, has_doc):
        """Validation step - returns predictions."""
        (
            total_loss,
            coords_pred,
            loss_simcc,
            loss_coord,
            loss_score,
            score_logit,
        ) = self._compute_loss(images, coords_gt, has_doc, training=False)
        return coords_pred, score_logit, total_loss, loss_simcc, loss_coord, loss_score


def compute_metrics(coords_pred, coords_gt, has_doc, img_size, score_logit=None):
    """Compute detailed IoU, error, and score metrics."""
    metrics = {}
    mask = has_doc > 0.5
    pos_count = int(mask.sum())
    neg_count = int((~mask).sum())
    metrics["pos_count"] = pos_count
    metrics["neg_count"] = neg_count

    if pos_count == 0:
        metrics.update({
            "mean_iou": 0.0,
            "median_iou": 0.0,
            "p90_iou": 0.0,
            "p95_iou": 0.0,
            "p99_iou": 0.0,
            "mean_err": 999.0,
            "median_err": 999.0,
            "p90_err": 999.0,
            "p95_err": 999.0,
            "p99_err": 999.0,
        })
    else:
        pred_pos = coords_pred[mask].reshape(-1, 4, 2)
        gt_pos = coords_gt[mask].reshape(-1, 4, 2)

        # Bounding box IoU (fast approximation)
        pred_min = pred_pos.min(axis=1)
        pred_max = pred_pos.max(axis=1)
        gt_min = gt_pos.min(axis=1)
        gt_max = gt_pos.max(axis=1)

        inter_min = np.maximum(pred_min, gt_min)
        inter_max = np.minimum(pred_max, gt_max)
        inter_wh = np.maximum(inter_max - inter_min, 0)
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]

        pred_area = (pred_max - pred_min).prod(axis=1)
        gt_area = (gt_max - gt_min).prod(axis=1)
        union_area = pred_area + gt_area - inter_area + 1e-9

        ious = inter_area / union_area
        metrics["mean_iou"] = float(ious.mean())
        metrics["median_iou"] = float(np.median(ious))
        metrics["p90_iou"] = float(np.percentile(ious, 90))
        metrics["p95_iou"] = float(np.percentile(ious, 95))
        metrics["p99_iou"] = float(np.percentile(ious, 99))

        errors = np.abs(pred_pos - gt_pos) * img_size
        error_per_sample = errors.reshape(errors.shape[0], -1).mean(axis=1)
        metrics["mean_err"] = float(error_per_sample.mean())
        metrics["median_err"] = float(np.median(error_per_sample))
        metrics["p90_err"] = float(np.percentile(error_per_sample, 90))
        metrics["p95_err"] = float(np.percentile(error_per_sample, 95))
        metrics["p99_err"] = float(np.percentile(error_per_sample, 99))

    if score_logit is not None:
        scores = 1.0 / (1.0 + np.exp(-np.clip(score_logit, -60.0, 60.0)))
        scores = scores.reshape(-1)
        labels = has_doc.reshape(-1)
        metrics["score_acc"] = float(((scores >= 0.5) == (labels >= 0.5)).mean())
        if pos_count > 0:
            metrics["score_pos_mean"] = float(scores[mask].mean())
        else:
            metrics["score_pos_mean"] = 0.0
        if neg_count > 0:
            metrics["score_neg_mean"] = float(scores[~mask].mean())
        else:
            metrics["score_neg_mean"] = 0.0

    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ultra-optimized cross-platform training")

    # Data
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root directory with images/, labels/, split files")
    parser.add_argument("--hf_dataset", type=str, default=None,
                        help="HuggingFace dataset name (e.g., 'mapo80/DocCornerDataset') "
                             "or path to local Parquet directory")
    parser.add_argument("--download_hf", type=str, default=None,
                        help="Download HuggingFace dataset to this directory and exit. "
                             "Use with --hf_dataset to specify the dataset name.")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace API token for private datasets. "
                             "Can also be set via HF_TOKEN environment variable.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for checkpoints (required unless --download_hf)")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="validation",
                        help="Validation split name (default: 'validation' for HF, 'val' for local)")
    parser.add_argument("--experiment_name", type=str, default=None)

    # Model
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenetv2",
        choices=["mobilenetv2", "mobilenetv3_small", "mobilenetv3_large", "cspnext"],
        help="Backbone architecture",
    )
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument(
        "--backbone_weights",
        type=str,
        default="imagenet",
        help="Backbone init weights ('imagenet' or None). None avoids downloads.",
    )
    parser.add_argument(
        "--init_weights",
        type=str,
        default=None,
        help=(
            "Optional warm-start weights (.weights.h5) or a checkpoint directory containing best_model.weights.h5. "
            "Useful for fine-tuning at a different img_size/num_bins."
        ),
    )
    parser.add_argument(
        "--init_partial",
        action="store_true",
        help="If strict init load fails, retry with by_name=True, skip_mismatch=True (HDF5 only).",
    )
    parser.add_argument("--fpn_ch", type=int, default=32)
    parser.add_argument("--simcc_ch", type=int, default=96)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_bins", type=int, default=256)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--use_gau", action="store_true",
                        help="Enable Gated Attention Unit for corner relationship modeling (RTMPose-inspired)")
    parser.add_argument("--gau_hidden_dim", type=int, default=64,
                        help="Hidden dimension for GAU self-attention")
    parser.add_argument("--fc_expansion_dim", type=int, default=0,
                        help="FC expansion dimension before classification (0=disabled, 256=RTMPose default)")

    # Loss
    parser.add_argument("--sigma_px", type=float, default=3.0)
    parser.add_argument("--w_simcc", type=float, default=1.0)
    parser.add_argument("--w_coord", type=float, default=0.5)
    parser.add_argument("--w_score", type=float, default=0.5)

    # Training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * accumulation_steps)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr_patience", type=int, default=7)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    # Loading
    parser.add_argument("--num_workers", type=int, default=64,
                        help="Threads for image loading")

    # Augmentation
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation")

    # Hard Mining (OHEM)
    parser.add_argument("--hard_mining", action="store_true",
                        help="Enable Online Hard Example Mining")
    parser.add_argument("--hard_mining_weight", type=float, default=2.0,
                        help="Extra weight for hard samples (total = 1 + this)")
    parser.add_argument("--hard_mining_threshold", type=float, default=20.0,
                        help="Corner error threshold (px) to classify as hard sample")
    parser.add_argument("--hard_mining_start", type=float, default=0.2,
                        help="Fraction of epochs before activating hard mining (curriculum)")

    args = parser.parse_args()

    # Handle download mode
    if args.download_hf:
        if not args.hf_dataset:
            parser.error("--download_hf requires --hf_dataset to specify the dataset name")
        download_hf_dataset(args.hf_dataset, args.download_hf, hf_token=args.hf_token)
        return

    # Validate data source arguments
    if args.data_root is None and args.hf_dataset is None:
        parser.error("Either --data_root or --hf_dataset must be specified")
    if args.data_root is not None and args.hf_dataset is not None:
        parser.error("Cannot specify both --data_root and --hf_dataset")
    if args.output_dir is None:
        parser.error("--output_dir is required for training")

    # Auto-detect val_split name for local datasets
    if args.data_root is not None and args.val_split == "validation":
        # Local datasets typically use 'val' instead of 'validation'
        args.val_split = "val"

    # Setup platform
    platform = setup_platform()
    use_mixed_precision = platform == 'cuda'

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.experiment_name:
        output_dir = Path(args.output_dir) / args.experiment_name
    else:
        output_dir = Path(args.output_dir) / f"{args.backbone}_{args.img_size}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}", flush=True)

    # Save config
    config = vars(args).copy()
    config["platform"] = platform
    config["mixed_precision"] = use_mixed_precision
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ========================================================================
    # Load datasets
    # ========================================================================
    print("\n" + "=" * 80, flush=True)
    print("Loading datasets...", flush=True)
    print("=" * 80, flush=True)

    if args.hf_dataset:
        # Load from HuggingFace (Hub or local Parquet)
        print(f"Data source: HuggingFace ({args.hf_dataset})", flush=True)
        train_images, train_coords, train_has_doc = load_hf_dataset_to_numpy(
            args.hf_dataset, args.train_split, args.img_size, args.num_workers
        )
        val_images, val_coords, val_has_doc = load_hf_dataset_to_numpy(
            args.hf_dataset, args.val_split, args.img_size, args.num_workers
        )
    else:
        # Load from local directory
        print(f"Data source: Local ({args.data_root})", flush=True)
        train_images, train_coords, train_has_doc = load_dataset_fast(
            args.data_root, args.train_split, args.img_size, args.num_workers
        )
        val_images, val_coords, val_has_doc = load_dataset_fast(
            args.data_root, args.val_split, args.img_size, args.num_workers
        )

    # ========================================================================
    # Create tf.data datasets
    # ========================================================================
    print("\n" + "=" * 80, flush=True)
    print("Creating tf.data pipelines...", flush=True)
    print("=" * 80, flush=True)

    print("Train dataset:", flush=True)
    train_ds = FastDataset(train_images, train_coords, train_has_doc,
                           args.batch_size, shuffle=True, drop_remainder=True, name="train",
                           return_indices=args.hard_mining)
    del train_images, train_coords, train_has_doc
    gc.collect()

    print("Val dataset:", flush=True)
    val_ds = FastDataset(val_images, val_coords, val_has_doc,
                         args.batch_size, shuffle=False, drop_remainder=False, name="val",
                         return_indices=args.hard_mining)
    del val_images, val_coords, val_has_doc
    gc.collect()

    # ========================================================================
    # Create model
    # ========================================================================
    print("\n" + "=" * 80, flush=True)
    print("Creating model...", flush=True)
    print("=" * 80, flush=True)

    model = create_model(
        backbone=args.backbone,
        alpha=args.alpha,
        backbone_weights=_normalize_backbone_weights(args.backbone_weights),
        fpn_ch=args.fpn_ch,
        simcc_ch=args.simcc_ch,
        img_size=args.img_size,
        num_bins=args.num_bins,
        tau=args.tau,
        use_gau=args.use_gau,
        gau_hidden_dim=args.gau_hidden_dim,
        fc_expansion_dim=args.fc_expansion_dim,
    )
    print(f"Parameters: {model.count_params():,}", flush=True)
    if args.use_gau:
        print(f"GAU enabled: hidden_dim={args.gau_hidden_dim}", flush=True)
    if args.fc_expansion_dim > 0:
        print(f"FC expansion: dim={args.fc_expansion_dim}", flush=True)

    # Optional warm-start (e.g. fine-tune at different img_size/num_bins).
    if args.init_weights:
        init_path = _find_init_weights_path(args.init_weights)
        print(f"\nLoading init weights from: {init_path}", flush=True)

        # Keras by_name loading only supports legacy HDF5 files ending in .h5/.hdf5.
        # Our checkpoints typically use the newer '*.weights.h5' naming; the file is
        # still HDF5, but Keras refuses by_name based on the filename.
        init_for_by_name = init_path
        if init_path.name.endswith(".weights.h5"):
            legacy_name = init_path.name[: -len(".weights.h5")] + ".h5"
            legacy_path = output_dir / legacy_name
            if not legacy_path.exists():
                shutil.copy2(init_path, legacy_path)
            init_for_by_name = legacy_path

        if args.init_partial:
            # Prefer partial load first to avoid noisy shape-mismatch errors when the
            # architecture changes slightly (e.g. simcc_ch, img_size/num_bins).
            try:
                model.load_weights(str(init_for_by_name), by_name=True, skip_mismatch=True)
                print("✓ Loaded init weights (partial)", flush=True)
            except Exception as e:
                print(f"Warning: partial init load failed: {e}", flush=True)
                print("Retrying strict init load...", flush=True)
                model.load_weights(str(init_path))
                print("✓ Loaded init weights (strict)", flush=True)
        else:
            model.load_weights(str(init_path))
            print("✓ Loaded init weights (strict)", flush=True)

    # Optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    if use_mixed_precision:
        from tensorflow.keras import mixed_precision
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    # Trainer
    trainer = Trainer(
        model, optimizer, args.img_size, args.sigma_px, args.tau,
        args.w_simcc, args.w_coord, args.w_score,
        platform=platform, augment=args.augment,
        accumulation_steps=args.accumulation_steps
    )

    # ========================================================================
    # Warmup (compile XLA kernels)
    # ========================================================================
    print("\nCompiling (warmup)...", flush=True)
    for batch in tqdm(
        train_ds.dataset.take(1),
        total=1,
        desc="Warmup",
        unit="batch",
    ):
        if args.hard_mining:
            _, images, coords, has_doc = batch
        else:
            images, coords, has_doc = batch
        _ = trainer.train_step(images, coords, has_doc)
        _ = trainer.val_step(images, coords, has_doc)
    print("Warmup done!", flush=True)

    # ========================================================================
    # Training loop
    # ========================================================================
    effective_batch_size = args.batch_size * args.accumulation_steps
    print("\n" + "=" * 80, flush=True)
    print(f"Starting training: {args.epochs} epochs", flush=True)
    if args.accumulation_steps > 1:
        print(f"Batch size: {args.batch_size} x {args.accumulation_steps} accumulation = {effective_batch_size} effective", flush=True)
    else:
        print(f"Batch size: {args.batch_size}", flush=True)
    if args.augment:
        print("Augmentation: ENABLED", flush=True)
    if args.hard_mining:
        print(f"Hard mining: ENABLED (threshold={args.hard_mining_threshold}px, weight={args.hard_mining_weight}, start={args.hard_mining_start*100:.0f}%)", flush=True)
    print("=" * 80, flush=True)

    best_iou = 0.0
    best_epoch = 0
    current_lr = args.lr
    no_improve_count = 0
    lr_no_improve_count = 0
    history = {"train": [], "val": []}

    # Initialize hard sample tracker for OHEM
    hard_tracker = None
    hard_mining_active = False
    hard_mining_start_epoch = int(args.epochs * args.hard_mining_start)
    if args.hard_mining:
        hard_tracker = HardSampleTracker(threshold_px=args.hard_mining_threshold)
        # Load hard samples from previous checkpoint if exists
        hard_samples_path = output_dir / "hard_samples.npz"
        if hard_samples_path.exists():
            hard_tracker.load(hard_samples_path)
            n_hard, avg_err = hard_tracker.get_stats()
            print(f"Loaded {n_hard} hard samples from checkpoint (avg err: {avg_err:.1f}px)", flush=True)

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Reshuffle training data for new epoch
        train_ds.reshuffle()

        # Warmup LR
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            optimizer.learning_rate.assign(warmup_lr)
            current_lr = warmup_lr

        # Activate hard mining after curriculum warmup
        if args.hard_mining and epoch >= hard_mining_start_epoch:
            hard_mining_active = True

        epoch_status = f"Epoch {epoch + 1}/{args.epochs}"
        if args.hard_mining:
            if hard_mining_active:
                n_hard, _ = hard_tracker.get_stats()
                epoch_status += f" [OHEM: {n_hard} hard samples]"
            else:
                epoch_status += f" [OHEM: inactive until epoch {hard_mining_start_epoch + 1}]"
        print(f"\n{epoch_status}", flush=True)

        # Training
        train_losses = []
        train_simcc = []
        train_coord = []
        train_score = []
        train_iou = []
        train_err = []
        accumulation_steps = args.accumulation_steps
        use_accumulation = accumulation_steps > 1

        # Initialize gradient accumulator on first use
        if use_accumulation and trainer._gradient_accumulator is None:
            trainer._init_gradient_accumulator()
            trainer.reset_gradient_accumulator()

        train_pbar = tqdm(
            train_ds.dataset,
            total=len(train_ds),
            desc="  Train",
            unit="batch",
            ncols=120,
            leave=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]",
            ascii=True,
        )
        batch_start_time = time.time()
        imgs_processed = 0
        for batch_idx, batch in enumerate(train_pbar):
            # Unpack batch (with or without indices)
            if args.hard_mining:
                indices, images, coords, has_doc = batch
            else:
                images, coords, has_doc = batch
                indices = None

            if args.augment:
                images, coords = trainer.augment_batch(images, coords, has_doc)

            # Compute sample weights for OHEM
            sample_weights = None
            if hard_mining_active and hard_tracker is not None and indices is not None:
                # Build sample weights: 1.0 for normal, 1.0 + hard_mining_weight for hard
                weights = np.ones(len(indices), dtype=np.float32)
                for i, idx in enumerate(indices.numpy()):
                    if hard_tracker.is_hard(idx):
                        weights[i] = 1.0 + args.hard_mining_weight
                sample_weights = tf.constant(weights, dtype=tf.float32)

            # Training step (weighted or standard)
            if sample_weights is not None:
                if use_accumulation:
                    loss, loss_simcc, loss_coord, loss_score, batch_iou, batch_err = trainer.train_step_accumulate_weighted(
                        images, coords, has_doc, sample_weights
                    )
                    if (batch_idx + 1) % accumulation_steps == 0:
                        trainer.apply_accumulated_gradients()
                else:
                    loss, loss_simcc, loss_coord, loss_score, batch_iou, batch_err = trainer.train_step_weighted(
                        images, coords, has_doc, sample_weights
                    )
            else:
                if use_accumulation:
                    loss, loss_simcc, loss_coord, loss_score, batch_iou, batch_err = trainer.train_step_accumulate(
                        images, coords, has_doc
                    )
                    if (batch_idx + 1) % accumulation_steps == 0:
                        trainer.apply_accumulated_gradients()
                else:
                    loss, loss_simcc, loss_coord, loss_score, batch_iou, batch_err = trainer.train_step(
                        images, coords, has_doc
                    )

            loss_val = float(loss)
            train_losses.append(loss_val)
            train_simcc.append(float(loss_simcc))
            train_coord.append(float(loss_coord))
            train_score.append(float(loss_score))
            train_iou.append(float(batch_iou))
            train_err.append(float(batch_err))

            # Calculate img/s
            imgs_processed += images.shape[0]
            elapsed = time.time() - batch_start_time
            imgs_per_sec = imgs_processed / elapsed if elapsed > 0 else 0

            train_pbar.set_postfix({
                "loss": f"{np.mean(train_losses):.4f}",
                "err": f"{np.mean(train_err):.1f}px",
                "iou": f"{np.mean(train_iou):.3f}",
                "img/s": f"{imgs_per_sec:.0f}",
            })

        # Apply any remaining accumulated gradients at end of epoch
        if use_accumulation and trainer._accumulation_count > 0:
            trainer.apply_accumulated_gradients()

        avg_train_loss = float(np.mean(train_losses))
        avg_train_simcc = float(np.mean(train_simcc))
        avg_train_coord = float(np.mean(train_coord))
        avg_train_score = float(np.mean(train_score))
        avg_train_iou = float(np.mean(train_iou)) if train_iou else 0.0
        avg_train_err = float(np.mean(train_err)) if train_err else 0.0

        # Validation
        val_losses = []
        val_simcc = []
        val_coord = []
        val_score = []
        metrics = ValidationMetrics(img_size=args.img_size)
        val_pbar = tqdm(
            val_ds.dataset,
            total=len(val_ds),
            desc="  Val  ",
            unit="batch",
            ncols=120,
            leave=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]",
            ascii=True,
        )
        val_start_time = time.time()
        val_imgs_processed = 0
        for batch in val_pbar:
            # Unpack batch (with or without indices)
            if args.hard_mining:
                indices, images, coords, has_doc = batch
            else:
                images, coords, has_doc = batch
                indices = None

            preds, score_logit, v_loss, v_simcc, v_coord, v_score = trainer.val_step(
                images, coords, has_doc
            )
            val_losses.append(float(v_loss))
            val_simcc.append(float(v_simcc))
            val_coord.append(float(v_coord))
            val_score.append(float(v_score))
            score_pred = tf.sigmoid(score_logit).numpy()

            # Update metrics (with or without indices)
            if args.hard_mining and indices is not None:
                metrics.update_with_indices(
                    indices.numpy(), preds.numpy(), coords.numpy(), score_pred, has_doc.numpy()
                )
            else:
                metrics.update(preds.numpy(), coords.numpy(), score_pred, has_doc.numpy())

            # Calculate img/s
            val_imgs_processed += images.shape[0]
            val_elapsed = time.time() - val_start_time
            val_imgs_per_sec = val_imgs_processed / val_elapsed if val_elapsed > 0 else 0

            val_pbar.set_postfix({
                "loss": f"{np.mean(val_losses):.4f}",
                "img/s": f"{val_imgs_per_sec:.0f}",
            })

        # Compute metrics and update hard sample tracker
        if args.hard_mining:
            val_metrics, pos_indices, max_corner_errors = metrics.compute_with_indices()
            # Update hard sample tracker with validation results
            hard_tracker.update_from_validation(pos_indices, max_corner_errors)
            n_hard, avg_err = hard_tracker.get_stats()
            # Save hard samples to checkpoint
            hard_tracker.save(output_dir / "hard_samples.npz")
        else:
            val_metrics = metrics.compute()

        epoch_time = time.time() - epoch_start
        samples_per_sec = (len(train_ds) * args.batch_size) / epoch_time

        # Logging
        avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        avg_val_simcc = float(np.mean(val_simcc)) if val_simcc else 0.0
        avg_val_coord = float(np.mean(val_coord)) if val_coord else 0.0
        avg_val_score = float(np.mean(val_score)) if val_score else 0.0

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | LR={current_lr:.1e} | "
            f"{epoch_time:.1f}s ({samples_per_sec:.0f} img/s)",
            flush=True,
        )
        print(
            f"    Train: loss={avg_train_loss:.4f} "
            f"(simcc={avg_train_simcc:.4f} coord={avg_train_coord:.4f} score={avg_train_score:.4f})",
            flush=True,
        )
        print(
            f"           err={avg_train_err:.1f}px  IoU={avg_train_iou:.3f}",
            flush=True,
        )
        print(
            f"    Val:   loss={avg_val_loss:.4f} "
            f"(simcc={avg_val_simcc:.4f} coord={avg_val_coord:.4f} score={avg_val_score:.4f})",
            flush=True,
        )
        print(
            f"           err_mean={val_metrics['corner_error_px']:.1f}px  "
            f"err_p95={val_metrics['corner_error_p95_px']:.1f}px  "
            f"err_max={val_metrics['corner_error_max_px']:.1f}px  "
            f"err_worst={val_metrics['corner_error_worst_px']:.1f}px",
            flush=True,
        )
        print(
            f"           IoU={val_metrics['mean_iou']:.4f}  "
            f"R@90={val_metrics['recall_90']*100:.1f}%  "
            f"R@95={val_metrics['recall_95']*100:.1f}%  "
            f"R@99={val_metrics['recall_99']*100:.1f}%",
            flush=True,
        )
        n_doc = int(val_metrics.get("num_with_doc", 0))
        if n_doc > 0:
            out_iou90 = int(val_metrics.get("num_iou_lt_90", 0))
            out_err20 = int(val_metrics.get("num_err_gt_20", 0))
            out_err50 = int(val_metrics.get("num_err_gt_50", 0))
            any_c20 = int(val_metrics.get("num_any_corner_gt_20", 0))
            any_c50 = int(val_metrics.get("num_any_corner_gt_50", 0))
            print(
                f"           outliers: IoU<0.90={out_iou90}/{n_doc} ({out_iou90/n_doc*100:.1f}%)  "
                f"err>20px={out_err20}/{n_doc} ({out_err20/n_doc*100:.1f}%)  "
                f"any_corner>20px={any_c20}/{n_doc} ({any_c20/n_doc*100:.1f}%)",
                flush=True,
            )
            if any_c50 > 0:
                print(
                    f"           any_corner>50px={any_c50}/{n_doc} ({any_c50/n_doc*100:.1f}%)",
                    flush=True,
                )
        print(
            f"           cls_acc={val_metrics['cls_accuracy']*100:.1f}%  "
            f"cls_f1={val_metrics['cls_f1']:.3f}",
            flush=True,
        )

        # Hard mining stats
        if args.hard_mining:
            n_hard, avg_hard_err = hard_tracker.get_stats()
            mining_status = "ACTIVE" if hard_mining_active else "inactive"
            print(
                f"           OHEM: {n_hard} hard samples (avg err: {avg_hard_err:.1f}px) [{mining_status}]",
                flush=True,
            )

        history["train"].append({
            "loss": avg_train_loss,
            "loss_simcc": avg_train_simcc,
            "loss_coord": avg_train_coord,
            "loss_score": avg_train_score,
            "iou": avg_train_iou,
            "corner_err_px": avg_train_err,
        })
        history["val"].append({
            "loss": avg_val_loss,
            "loss_simcc": avg_val_simcc,
            "loss_coord": avg_val_coord,
            "loss_score": avg_val_score,
            **val_metrics,
        })

        # Checkpointing
        if val_metrics["mean_iou"] > best_iou:
            best_iou = val_metrics["mean_iou"]
            best_epoch = epoch + 1
            no_improve_count = 0
            lr_no_improve_count = 0

            model.save_weights(str(output_dir / "best_model.weights.h5"))
            inference_model = create_inference_model(model)
            inference_model.save(output_dir / "best_model_inference.keras")

            print(f"  * New best IoU: {best_iou:.4f}", flush=True)
        else:
            no_improve_count += 1
            lr_no_improve_count += 1

            if epoch >= args.warmup_epochs:
                if lr_no_improve_count >= args.lr_patience and current_lr > args.min_lr:
                    current_lr = max(current_lr * args.lr_factor, args.min_lr)
                    optimizer.learning_rate.assign(current_lr)
                    lr_no_improve_count = 0
                    print(f"  -> Reduced LR to {current_lr:.2e}", flush=True)

        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if no_improve_count >= args.patience:
            print(f"\nEarly stopping at epoch {epoch + 1}", flush=True)
            break

    model.save_weights(str(output_dir / "final_model.weights.h5"))

    print("\n" + "=" * 80, flush=True)
    print("Training Complete!", flush=True)
    print("=" * 80, flush=True)
    print(f"Best epoch: {best_epoch} with IoU: {best_iou:.4f}", flush=True)
    print(f"Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
