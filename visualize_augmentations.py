"""
Visualize TensorFlow GPU augmentations vs original images.

Creates a grid showing original images with GT corners and
augmented images with transformed corners.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from dataset import (
    load_split_file,
    load_yolo_label,
    preload_images_to_cache,
    tf_augment_batch,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def denormalize_image(img_normalized):
    """Convert normalized image back to [0, 255] uint8."""
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img = img_normalized * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def draw_corners_on_image(img_array, coords, color=(255, 0, 0), width=2):
    """
    Draw quadrilateral corners on image.

    Args:
        img_array: numpy array [H, W, 3] uint8
        coords: [8] normalized coordinates (x0,y0,x1,y1,x2,y2,x3,y3)
        color: RGB tuple
        width: Line width

    Returns:
        PIL Image with corners drawn
    """
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)

    h, w = img_array.shape[:2]

    # Convert normalized coords to pixel coords
    points = [
        (coords[0] * w, coords[1] * h),  # TL
        (coords[2] * w, coords[3] * h),  # TR
        (coords[4] * w, coords[5] * h),  # BR
        (coords[6] * w, coords[7] * h),  # BL
    ]

    # Draw quadrilateral
    for i in range(4):
        p1 = points[i]
        p2 = points[(i + 1) % 4]
        draw.line([p1, p2], fill=color, width=width)

    # Draw corner points
    for i, (x, y) in enumerate(points):
        r = 4
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
        # Label corners
        labels = ["TL", "TR", "BR", "BL"]
        draw.text((x+5, y-10), labels[i], fill=color)

    return img


def visualize_tf_augmentations(
    data_root: str,
    output_dir: str,
    num_samples: int = 16,
    num_aug_versions: int = 3,
    split: str = "train",
    img_size: int = 224,
):
    """
    Create visualization of TensorFlow GPU augmentations.

    Args:
        data_root: Path to dataset root
        output_dir: Path to save visualizations
        num_samples: Number of samples to visualize
        num_aug_versions: Number of augmented versions per sample
        split: Split to use
        img_size: Image size
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find split file
    split_file = None
    for prefix in [f"{split}_with_negative_v2", f"{split}_with_negative", split]:
        candidate = data_root / f"{prefix}.txt"
        if candidate.exists():
            split_file = candidate
            break

    if split_file is None:
        raise FileNotFoundError(f"No split file found for {split}")

    image_list = load_split_file(str(split_file))

    # Filter to positive samples only
    positive_images = [name for name in image_list if not name.startswith("negative_")]

    # Sample images
    np.random.seed(42)
    selected = np.random.choice(positive_images, min(num_samples, len(positive_images)), replace=False)

    print(f"Loading {len(selected)} images...")

    # Load images and labels
    image_dir = data_root / "images"
    label_dir = data_root / "labels"

    images = []
    coords_list = []
    has_doc_list = []
    names = []

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)

    for name in selected:
        # Load image
        img_path = image_dir / name
        if not img_path.exists():
            # Try common extensions
            stem = Path(name).stem
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                candidate = image_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        img = img.resize((img_size, img_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_normalized = (img_np - mean) / std

        # Load label
        label_path = label_dir / f"{Path(name).stem}.txt"
        coords, has_doc = load_yolo_label(str(label_path))

        if has_doc:
            images.append(img_normalized)
            coords_list.append(coords)
            has_doc_list.append(1.0)
            names.append(name)

    print(f"Loaded {len(images)} valid images")

    # Convert to tensors
    images_tensor = tf.constant(np.array(images), dtype=tf.float32)
    coords_tensor = tf.constant(np.array(coords_list), dtype=tf.float32)
    has_doc_tensor = tf.constant(np.array(has_doc_list), dtype=tf.float32)

    # Create visualization grid
    n_samples = len(images)
    cols = 1 + num_aug_versions  # Original + augmented versions

    fig, axes = plt.subplots(n_samples, cols, figsize=(4*cols, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    print("Generating augmentations...")

    for i in range(n_samples):
        # Original image
        orig_img = denormalize_image(images[i])
        orig_coords = coords_list[i]

        orig_with_corners = draw_corners_on_image(orig_img, orig_coords, color=(0, 255, 0), width=2)

        axes[i, 0].imshow(orig_with_corners)
        axes[i, 0].set_title(f"Original\n{names[i][:30]}", fontsize=8)
        axes[i, 0].axis('off')

        # Augmented versions
        for j in range(num_aug_versions):
            # Run augmentation
            aug_images, aug_coords = tf_augment_batch(
                images_tensor[i:i+1],
                coords_tensor[i:i+1],
                has_doc_tensor[i:i+1],
                img_size
            )

            aug_img = denormalize_image(aug_images[0].numpy())
            aug_coord = aug_coords[0].numpy()

            aug_with_corners = draw_corners_on_image(aug_img, aug_coord, color=(255, 0, 0), width=2)

            axes[i, j+1].imshow(aug_with_corners)
            axes[i, j+1].set_title(f"Aug {j+1}", fontsize=8)
            axes[i, j+1].axis('off')

    plt.tight_layout()

    # Save
    output_path = output_dir / "tf_augmentations_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to {output_path}")

    # Also save individual comparison images
    print("Saving individual comparisons...")
    for i in range(min(5, n_samples)):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Original
        orig_img = denormalize_image(images[i])
        orig_coords = coords_list[i]
        orig_with_corners = draw_corners_on_image(orig_img, orig_coords, color=(0, 255, 0), width=3)

        axes[0].imshow(orig_with_corners)
        axes[0].set_title("Original (green)", fontsize=10)
        axes[0].axis('off')

        # 3 augmented versions
        for j in range(3):
            aug_images, aug_coords = tf_augment_batch(
                images_tensor[i:i+1],
                coords_tensor[i:i+1],
                has_doc_tensor[i:i+1],
                img_size
            )

            aug_img = denormalize_image(aug_images[0].numpy())
            aug_coord = aug_coords[0].numpy()

            aug_with_corners = draw_corners_on_image(aug_img, aug_coord, color=(255, 0, 0), width=3)

            axes[j+1].imshow(aug_with_corners)
            axes[j+1].set_title(f"Augmented {j+1} (red)", fontsize=10)
            axes[j+1].axis('off')

        plt.tight_layout()
        output_path = output_dir / f"sample_{i+1}_{names[i][:20]}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {output_path.name}")

    print("\nDone!")


if __name__ == "__main__":
    # Default paths
    data_root = "/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-labeled"
    output_dir = "/Volumes/ZX20/ML-Models/DocScannerDetection/models/DocCornerNetV3/visualization"

    # Check if data exists
    if not Path(data_root).exists():
        print(f"Dataset not found at {data_root}")
        print("Please update data_root path")
        sys.exit(1)

    visualize_tf_augmentations(
        data_root=data_root,
        output_dir=output_dir,
        num_samples=8,
        num_aug_versions=3,
        split="train",
        img_size=224,
    )
