#!/usr/bin/env python3
"""
Visualize all images with small bounding boxes, organized by prefix.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from collections import defaultdict


def load_yolo_obb_label(label_path: str):
    """Load YOLO OBB label."""
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


def compute_polygon_area(coords):
    """Compute area using Shoelace formula."""
    x = coords[0::2]
    y = coords[1::2]
    n = len(x)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j]
        area -= x[j] * y[i]
    return abs(area) / 2.0


def draw_bbox_on_image(image_path, coords, output_path, info_text=""):
    """Draw bbox and save image."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Convert normalized coords to pixels
    points = []
    for i in range(0, 8, 2):
        px = int(coords[i] * w)
        py = int(coords[i+1] * h)
        points.append((px, py))
    points.append(points[0])

    draw = ImageDraw.Draw(img)
    draw.line(points, fill=(255, 0, 0), width=3)

    # Draw corners
    for i, (px, py) in enumerate(points[:-1]):
        radius = 6
        draw.ellipse([px-radius, py-radius, px+radius, py+radius],
                     fill=(0, 255, 0), outline=(0, 200, 0))

    # Add info text at top
    if info_text:
        draw.rectangle([0, 0, w, 25], fill=(0, 0, 0))
        draw.text((5, 5), info_text, fill=(255, 255, 255))

    img.save(output_path)


def get_prefix(filename):
    """Extract prefix from filename."""
    # Handle different naming patterns
    parts = filename.split('_')
    if len(parts) >= 2:
        # Check for patterns like "id_detections", "id_card", "card_corner", etc.
        if parts[0] in ['id', 'card', 'idcard']:
            if len(parts) >= 3 and parts[1] in ['detections', 'card', 'corner', '4', 'final', 'jj', 'g2wbl', 'skew']:
                return f"{parts[0]}_{parts[1]}"
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def main():
    dataset_root = Path("/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new")
    output_base = Path("/Volumes/ZX20/ML-Models/DocCornerNet-CoordClass/all_small_bbox")
    output_base.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    area_threshold = 0.05
    dim_threshold = 0.1

    # Get all label files
    label_files = list(labels_dir.glob("*.txt"))
    print(f"Found {len(label_files)} label files")

    # Group by prefix
    small_by_prefix = defaultdict(list)

    for label_file in tqdm(label_files, desc="Scanning labels"):
        coords, has_doc = load_yolo_obb_label(str(label_file))
        if not has_doc:
            continue

        x = coords[0::2]
        y = coords[1::2]
        width = x.max() - x.min()
        height = y.max() - y.min()
        area = compute_polygon_area(coords)

        is_small = area < area_threshold or width < dim_threshold or height < dim_threshold

        if is_small:
            image_stem = label_file.stem
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                candidate = images_dir / f"{image_stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path:
                prefix = get_prefix(image_path.name)
                small_by_prefix[prefix].append({
                    'image_name': image_path.name,
                    'image_path': str(image_path),
                    'coords': coords,
                    'width': width,
                    'height': height,
                    'area': area
                })

    # Sort each group by area
    for prefix in small_by_prefix:
        small_by_prefix[prefix].sort(key=lambda x: x['area'])

    # Print summary
    print("\nSummary by prefix:")
    print("-" * 60)
    for prefix, images in sorted(small_by_prefix.items(), key=lambda x: -len(x[1])):
        print(f"{prefix}: {len(images)} images")

    # Generate visualizations for each prefix
    print("\nGenerating visualizations...")

    for prefix, images in small_by_prefix.items():
        prefix_dir = output_base / prefix
        prefix_dir.mkdir(parents=True, exist_ok=True)

        # Limit to first 100 per prefix to avoid too many files
        images_to_process = images[:100]

        for info in tqdm(images_to_process, desc=f"Drawing {prefix}", leave=False):
            info_text = f"A:{info['area']*100:.1f}% W:{info['width']*100:.0f}% H:{info['height']*100:.0f}%"
            output_file = prefix_dir / info['image_name']
            draw_bbox_on_image(info['image_path'], info['coords'], str(output_file), info_text)

        # Save list for this prefix
        list_file = prefix_dir / "image_list.txt"
        with open(list_file, 'w') as f:
            for info in images:
                f.write(f"{info['image_name']}\t{info['area']:.6f}\n")

    print(f"\nAll visualizations saved to: {output_base}")
    print(f"Each prefix has its own subfolder with up to 100 images")


if __name__ == "__main__":
    main()
