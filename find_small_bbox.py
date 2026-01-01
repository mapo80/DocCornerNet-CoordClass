#!/usr/bin/env python3
"""
Find images with small bounding boxes in the dataset.

A bbox is considered "small" when its area (relative to image size)
is below a threshold.
"""

import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw
import argparse
from tqdm import tqdm


def load_yolo_obb_label(label_path: str) -> Tuple[np.ndarray, bool]:
    """
    Load coordinates from YOLO OBB format label file.
    Format: class_id x0 y0 x1 y1 x2 y2 x3 y3 (normalized 0-1)
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


def compute_polygon_area(coords: np.ndarray) -> float:
    """
    Compute area of quadrilateral using Shoelace formula.
    coords: [x0, y0, x1, y1, x2, y2, x3, y3] normalized 0-1
    Returns area as fraction of image (0-1).
    """
    # Extract points
    x = coords[0::2]  # [x0, x1, x2, x3]
    y = coords[1::2]  # [y0, y1, y2, y3]

    # Shoelace formula
    n = len(x)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j]
        area -= x[j] * y[i]
    area = abs(area) / 2.0

    return area


def compute_bbox_dimensions(coords: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute bounding box dimensions.
    Returns (width, height, area) all normalized 0-1.
    """
    x = coords[0::2]
    y = coords[1::2]

    width = x.max() - x.min()
    height = y.max() - y.min()

    # Use polygon area for actual quadrilateral
    area = compute_polygon_area(coords)

    return width, height, area


def draw_bbox_on_image(image_path: str, coords: np.ndarray, output_path: str):
    """Draw the bounding box polygon on the image and save."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Convert normalized coords to pixel coords
    points = []
    for i in range(0, 8, 2):
        px = int(coords[i] * w)
        py = int(coords[i+1] * h)
        points.append((px, py))

    # Close the polygon
    points.append(points[0])

    draw = ImageDraw.Draw(img)
    # Draw polygon outline
    draw.line(points, fill=(255, 0, 0), width=3)

    # Draw corner points
    for i, (px, py) in enumerate(points[:-1]):
        radius = 5
        draw.ellipse([px-radius, py-radius, px+radius, py+radius],
                     fill=(0, 255, 0), outline=(0, 200, 0))
        # Label corners
        draw.text((px+8, py-8), f"{i}", fill=(255, 255, 0))

    img.save(output_path)
    return img


def find_small_bbox_images(
    dataset_root: str,
    area_threshold: float = 0.05,  # 5% of image area
    min_dimension_threshold: float = 0.1,  # 10% of image dimension
    output_dir: str = None,
    max_images: int = None
) -> List[dict]:
    """
    Find images where the GT bbox is too small.

    Args:
        dataset_root: Path to dataset with images/ and labels/ folders
        area_threshold: Minimum area as fraction of image (0-1)
        min_dimension_threshold: Minimum width/height as fraction (0-1)
        output_dir: Directory to save images with drawn bboxes
        max_images: Maximum number of small bbox images to process

    Returns:
        List of dicts with image info
    """
    dataset_path = Path(dataset_root)
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Create output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Get all label files
    label_files = list(labels_dir.glob("*.txt"))
    print(f"Found {len(label_files)} label files")

    small_bbox_images = []

    for label_file in tqdm(label_files, desc="Scanning labels"):
        coords, has_doc = load_yolo_obb_label(str(label_file))

        if not has_doc:
            continue

        width, height, area = compute_bbox_dimensions(coords)

        # Check if bbox is too small
        is_small = False
        reason = []

        if area < area_threshold:
            is_small = True
            reason.append(f"area={area:.4f} < {area_threshold}")

        if width < min_dimension_threshold:
            is_small = True
            reason.append(f"width={width:.4f} < {min_dimension_threshold}")

        if height < min_dimension_threshold:
            is_small = True
            reason.append(f"height={height:.4f} < {min_dimension_threshold}")

        if is_small:
            # Find corresponding image
            image_stem = label_file.stem
            image_path = None

            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                candidate = images_dir / f"{image_stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path:
                info = {
                    'image_name': image_path.name,
                    'image_path': str(image_path),
                    'label_path': str(label_file),
                    'coords': coords,
                    'width': width,
                    'height': height,
                    'area': area,
                    'reason': ', '.join(reason)
                }
                small_bbox_images.append(info)

                # Draw and save if output_dir specified
                if output_dir and (max_images is None or len(small_bbox_images) <= max_images):
                    output_file = output_path / f"small_{image_path.name}"
                    draw_bbox_on_image(str(image_path), coords, str(output_file))

    # Sort by area (smallest first)
    small_bbox_images.sort(key=lambda x: x['area'])

    return small_bbox_images


def main():
    parser = argparse.ArgumentParser(description='Find images with small bounding boxes')
    parser.add_argument('--dataset', type=str,
                        default='/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new',
                        help='Dataset root directory')
    parser.add_argument('--area-threshold', type=float, default=0.05,
                        help='Minimum bbox area as fraction of image (default: 0.05 = 5%%)')
    parser.add_argument('--dim-threshold', type=float, default=0.1,
                        help='Minimum width/height as fraction (default: 0.1 = 10%%)')
    parser.add_argument('--output-dir', type=str, default='small_bbox_output',
                        help='Directory to save annotated images')
    parser.add_argument('--max-images', type=int, default=50,
                        help='Maximum number of images to save')

    args = parser.parse_args()

    print(f"Scanning dataset: {args.dataset}")
    print(f"Area threshold: {args.area_threshold} ({args.area_threshold*100:.1f}%)")
    print(f"Dimension threshold: {args.dim_threshold} ({args.dim_threshold*100:.1f}%)")

    small_images = find_small_bbox_images(
        dataset_root=args.dataset,
        area_threshold=args.area_threshold,
        min_dimension_threshold=args.dim_threshold,
        output_dir=args.output_dir,
        max_images=args.max_images
    )

    print(f"\nFound {len(small_images)} images with small bboxes")

    if small_images:
        print("\nTop 20 smallest bboxes:")
        print("-" * 80)
        for i, info in enumerate(small_images[:20]):
            print(f"{i+1:3d}. {info['image_name']}")
            print(f"     Area: {info['area']*100:.2f}%, Width: {info['width']*100:.1f}%, Height: {info['height']*100:.1f}%")
            print(f"     Reason: {info['reason']}")

        print(f"\nAnnotated images saved to: {args.output_dir}/")

        # Save list to file
        list_file = Path(args.output_dir) / "small_bbox_list.txt"
        with open(list_file, 'w') as f:
            for info in small_images:
                f.write(f"{info['image_name']}\t{info['area']:.6f}\t{info['reason']}\n")
        print(f"Full list saved to: {list_file}")


if __name__ == "__main__":
    main()
