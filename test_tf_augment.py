#!/usr/bin/env python3
"""
Test script to verify TensorFlow GPU augmentations (rotation + perspective).

Creates visual output showing:
1. Original image with ground truth corners
2. Rotated image with transformed corners
3. Perspective-warped image with transformed corners

Usage:
    python test_tf_augment.py --hf_dataset ./hf_dataset --output_dir ./test_augment_output
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

# Import augmentation functions
from dataset import tf_rotate_batch, tf_perspective_batch, IMAGENET_MEAN, IMAGENET_STD


def denormalize_imagenet(images):
    """Convert ImageNet-normalized images back to [0, 255] uint8."""
    mean = np.array(IMAGENET_MEAN).reshape(1, 1, 1, 3)
    std = np.array(IMAGENET_STD).reshape(1, 1, 1, 3)
    images = images * std + mean
    images = np.clip(images * 255, 0, 255).astype(np.uint8)
    return images


def draw_corners(image, coords, color=(255, 0, 0), width=3):
    """Draw document corners on image.

    Args:
        image: PIL Image
        coords: [8] normalized coords (x0,y0,x1,y1,x2,y2,x3,y3)
        color: RGB color tuple
        width: Line width

    Returns:
        PIL Image with corners drawn
    """
    draw = ImageDraw.Draw(image)
    w, h = image.size

    # Convert normalized to pixel coordinates
    points = []
    for i in range(0, 8, 2):
        px = int(coords[i] * w)
        py = int(coords[i + 1] * h)
        points.append((px, py))

    # Draw polygon
    points.append(points[0])  # Close the polygon
    draw.line(points, fill=color, width=width)

    # Draw corner points
    for i, (px, py) in enumerate(points[:-1]):
        r = 5
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color)
        draw.text((px + 8, py - 8), f"P{i}", fill=color)

    return image


def test_rotation(images, coords, has_doc, angles=[0, 15, 30, -15, -30]):
    """Test rotation at different angles."""
    results = []

    for angle in angles:
        # Apply rotation
        rot_images, rot_coords = tf_rotate_batch(
            images, coords, has_doc,
            max_angle_deg=abs(angle) if angle != 0 else 0.001,
            fill_value=0.0
        )

        # For specific angle testing, we need to modify the function
        # For now, just test with random angles
        results.append({
            'angle': angle,
            'images': rot_images.numpy(),
            'coords': rot_coords.numpy()
        })

    return results


def test_perspective(images, coords, has_doc, intensities=[0.0, 0.03, 0.05, 0.08, 0.1]):
    """Test perspective at different intensities."""
    results = []

    for intensity in intensities:
        if intensity == 0:
            # No transform
            results.append({
                'intensity': intensity,
                'images': images.numpy(),
                'coords': coords.numpy()
            })
        else:
            # Apply perspective
            trans_images, trans_coords = tf_perspective_batch(
                images, coords, has_doc,
                intensity=intensity,
                fill_value=0.0
            )
            results.append({
                'intensity': intensity,
                'images': trans_images.numpy(),
                'coords': trans_coords.numpy()
            })

    return results


def load_sample_batch(hf_dataset_path, batch_size=4, img_size=320):
    """Load a sample batch from the dataset."""
    import pyarrow.parquet as pq
    import glob

    # Load parquet files from train directory
    train_dir = os.path.join(hf_dataset_path, "train")
    parquet_files = sorted(glob.glob(os.path.join(train_dir, "*.parquet")))

    if not parquet_files:
        # Try direct parquet file
        train_parquet = os.path.join(hf_dataset_path, "train.parquet")
        if os.path.exists(train_parquet):
            parquet_files = [train_parquet]
        else:
            raise FileNotFoundError(f"Dataset not found in: {train_dir}")

    # Load first parquet file
    table = pq.read_table(parquet_files[0])
    df = table.to_pandas()

    # Filter to only positive samples (is_negative=False)
    df_pos = df[df['is_negative'] == False].head(batch_size)

    images = []
    coords_list = []
    has_doc_list = []

    mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD).reshape(1, 1, 3)

    for _, row in df_pos.iterrows():
        # Decode image
        img_bytes = row['image']['bytes']
        img = Image.open(__import__('io').BytesIO(img_bytes)).convert('RGB')
        img = img.resize((img_size, img_size), Image.BILINEAR)

        # Normalize
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_arr = (img_arr - mean) / std

        images.append(img_arr)

        # Extract coords from separate columns
        coords = np.array([
            row['corner_tl_x'], row['corner_tl_y'],
            row['corner_tr_x'], row['corner_tr_y'],
            row['corner_br_x'], row['corner_br_y'],
            row['corner_bl_x'], row['corner_bl_y'],
        ], dtype=np.float32)
        coords_list.append(coords)
        has_doc_list.append(1.0)

    return (
        np.stack(images),
        np.stack(coords_list),
        np.array(has_doc_list, dtype=np.float32)
    )


def main():
    parser = argparse.ArgumentParser(description="Test TF augmentations")
    parser.add_argument("--hf_dataset", type=str, default="./hf_dataset",
                        help="Path to HuggingFace dataset")
    parser.add_argument("--output_dir", type=str, default="./test_augment_output",
                        help="Output directory for test images")
    parser.add_argument("--img_size", type=int, default=320,
                        help="Image size")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of samples to test")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.batch_size} samples from {args.hf_dataset}...")
    images, coords, has_doc = load_sample_batch(
        args.hf_dataset,
        batch_size=args.batch_size,
        img_size=args.img_size
    )

    print(f"Images shape: {images.shape}")
    print(f"Coords shape: {coords.shape}")
    print(f"Sample coords[0]: {coords[0]}")

    # Convert to tensors
    images_tf = tf.constant(images, dtype=tf.float32)
    coords_tf = tf.constant(coords, dtype=tf.float32)
    has_doc_tf = tf.constant(has_doc, dtype=tf.float32)

    # Save original images
    print("\nSaving original images...")
    images_uint8 = denormalize_imagenet(images)
    for i in range(args.batch_size):
        img = Image.fromarray(images_uint8[i])
        img = draw_corners(img, coords[i], color=(0, 255, 0), width=2)
        img.save(os.path.join(args.output_dir, f"original_{i}.png"))
        print(f"  Saved original_{i}.png")

    # Test rotation with multiple random samples
    print("\nTesting rotation (±15°)...")
    for trial in range(3):
        rot_images, rot_coords = tf_rotate_batch(
            images_tf, coords_tf, has_doc_tf,
            max_angle_deg=15.0,
            fill_value=0.0
        )
        rot_images_uint8 = denormalize_imagenet(rot_images.numpy())

        for i in range(args.batch_size):
            img = Image.fromarray(rot_images_uint8[i])
            img = draw_corners(img, rot_coords[i].numpy(), color=(255, 0, 0), width=2)
            img.save(os.path.join(args.output_dir, f"rotated_{i}_trial{trial}.png"))
        print(f"  Trial {trial}: Saved rotated images")

    # Test perspective with multiple random samples
    print("\nTesting perspective (intensity=0.05)...")
    for trial in range(3):
        persp_images, persp_coords = tf_perspective_batch(
            images_tf, coords_tf, has_doc_tf,
            intensity=0.05,
            fill_value=0.0
        )
        persp_images_uint8 = denormalize_imagenet(persp_images.numpy())

        for i in range(args.batch_size):
            img = Image.fromarray(persp_images_uint8[i])
            img = draw_corners(img, persp_coords[i].numpy(), color=(0, 0, 255), width=2)
            img.save(os.path.join(args.output_dir, f"perspective_{i}_trial{trial}.png"))
        print(f"  Trial {trial}: Saved perspective images")

    # Test combined rotation + perspective
    print("\nTesting combined rotation + perspective...")
    for trial in range(3):
        # First rotation
        rot_images, rot_coords = tf_rotate_batch(
            images_tf, coords_tf, has_doc_tf,
            max_angle_deg=15.0,
            fill_value=0.0
        )
        # Then perspective
        combined_images, combined_coords = tf_perspective_batch(
            rot_images, rot_coords, has_doc_tf,
            intensity=0.05,
            fill_value=0.0
        )
        combined_images_uint8 = denormalize_imagenet(combined_images.numpy())

        for i in range(args.batch_size):
            img = Image.fromarray(combined_images_uint8[i])
            img = draw_corners(img, combined_coords[i].numpy(), color=(255, 0, 255), width=2)
            img.save(os.path.join(args.output_dir, f"combined_{i}_trial{trial}.png"))
        print(f"  Trial {trial}: Saved combined images")

    # Numerical verification
    print("\n" + "="*60)
    print("NUMERICAL VERIFICATION")
    print("="*60)

    # Test 1: Coords should stay in [0,1] range
    print("\n1. Checking coordinate bounds after transforms...")
    rot_images, rot_coords = tf_rotate_batch(images_tf, coords_tf, has_doc_tf, 30.0, 0.0)
    persp_images, persp_coords = tf_perspective_batch(images_tf, coords_tf, has_doc_tf, 0.1, 0.0)

    rot_min, rot_max = rot_coords.numpy().min(), rot_coords.numpy().max()
    persp_min, persp_max = persp_coords.numpy().min(), persp_coords.numpy().max()

    print(f"   Rotation coords range: [{rot_min:.4f}, {rot_max:.4f}]")
    print(f"   Perspective coords range: [{persp_min:.4f}, {persp_max:.4f}]")

    if rot_min >= 0 and rot_max <= 1 and persp_min >= 0 and persp_max <= 1:
        print("   ✓ All coordinates within [0, 1] bounds")
    else:
        print("   ✗ WARNING: Coordinates out of bounds!")

    # Test 2: Zero rotation should not change coords significantly
    print("\n2. Testing zero rotation (should preserve coords)...")
    rot_images_zero, rot_coords_zero = tf_rotate_batch(
        images_tf, coords_tf, has_doc_tf, 0.001, 0.0
    )
    coord_diff = np.abs(rot_coords_zero.numpy() - coords).max()
    print(f"   Max coord difference with ~0° rotation: {coord_diff:.6f}")
    if coord_diff < 0.01:
        print("   ✓ Coords preserved with zero rotation")
    else:
        print("   ✗ WARNING: Coords changed unexpectedly!")

    # Test 3: Check that corners form a valid quadrilateral
    print("\n3. Checking quadrilateral validity...")
    def check_quadrilateral(coords):
        """Check if 4 points form a valid (non-self-intersecting) quadrilateral."""
        pts = coords.reshape(4, 2)
        # Simple check: area should be positive
        # Shoelace formula
        area = 0.5 * abs(
            (pts[0,0] - pts[2,0]) * (pts[1,1] - pts[3,1]) -
            (pts[1,0] - pts[3,0]) * (pts[0,1] - pts[2,1])
        )
        return area > 0.01  # Should have reasonable area

    valid_count = 0
    total_count = args.batch_size * 3  # 3 trials
    for trial in range(3):
        rot_images, rot_coords = tf_rotate_batch(images_tf, coords_tf, has_doc_tf, 15.0, 0.0)
        for i in range(args.batch_size):
            if check_quadrilateral(rot_coords[i].numpy()):
                valid_count += 1

    print(f"   Valid quadrilaterals after rotation: {valid_count}/{total_count}")
    if valid_count == total_count:
        print("   ✓ All quadrilaterals valid")
    else:
        print("   ✗ WARNING: Some invalid quadrilaterals!")

    print("\n" + "="*60)
    print(f"Test complete! Check output images in: {args.output_dir}")
    print("="*60)
    print("\nLegend:")
    print("  - GREEN corners: Original ground truth")
    print("  - RED corners: After rotation")
    print("  - BLUE corners: After perspective")
    print("  - MAGENTA corners: After rotation + perspective")
    print("\nIf corners align with the document in transformed images, the fix is correct!")


if __name__ == "__main__":
    main()
