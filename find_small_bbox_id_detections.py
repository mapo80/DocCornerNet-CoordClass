#!/usr/bin/env python3
"""
Find and visualize id_detections images with small bounding boxes.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


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
        draw.text((px+10, py-10), f"{i}", fill=(255, 255, 0))

    # Add info text
    if info_text:
        draw.rectangle([0, 0, w, 30], fill=(0, 0, 0))
        draw.text((5, 5), info_text, fill=(255, 255, 255))

    img.save(output_path)


def main():
    dataset_root = Path("/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new")
    output_dir = Path("/Volumes/ZX20/ML-Models/DocCornerNet-CoordClass/id_detections_small_bbox")
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    area_threshold = 0.05
    dim_threshold = 0.1

    # Find all id_detections labels
    label_files = list(labels_dir.glob("id_detections_*.txt"))
    print(f"Found {len(label_files)} id_detections label files")

    small_bbox_images = []

    for label_file in tqdm(label_files, desc="Scanning id_detections"):
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
                small_bbox_images.append({
                    'image_name': image_path.name,
                    'image_path': str(image_path),
                    'coords': coords,
                    'width': width,
                    'height': height,
                    'area': area
                })

    # Sort by area
    small_bbox_images.sort(key=lambda x: x['area'])

    print(f"\nFound {len(small_bbox_images)} id_detections with small bboxes")
    print("\nGenerating visualizations...")

    for info in tqdm(small_bbox_images, desc="Drawing bboxes"):
        info_text = f"{info['image_name']} | Area: {info['area']*100:.1f}% | W: {info['width']*100:.1f}% | H: {info['height']*100:.1f}%"
        output_file = output_dir / f"small_{info['image_name']}"
        draw_bbox_on_image(info['image_path'], info['coords'], str(output_file), info_text)

    print(f"\nImages saved to: {output_dir}")

    # Print summary
    print("\nSummary of small bbox id_detections:")
    print("-" * 80)
    for i, info in enumerate(small_bbox_images):
        print(f"{i+1:3d}. {info['image_name']}")
        print(f"     Area: {info['area']*100:.2f}%, Width: {info['width']*100:.1f}%, Height: {info['height']*100:.1f}%")

    # Save list
    list_file = output_dir / "id_detections_small_bbox_list.txt"
    with open(list_file, 'w') as f:
        for info in small_bbox_images:
            f.write(f"{info['image_name']}\n")
    print(f"\nList saved to: {list_file}")


if __name__ == "__main__":
    main()
