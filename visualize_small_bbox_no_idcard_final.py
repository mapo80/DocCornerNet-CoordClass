#!/usr/bin/env python3
"""
Visualize all images with small bounding boxes EXCEPT idcard_final.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


def load_yolo_obb_label(label_path: str):
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
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    points = []
    for i in range(0, 8, 2):
        px = int(coords[i] * w)
        py = int(coords[i+1] * h)
        points.append((px, py))
    points.append(points[0])

    draw = ImageDraw.Draw(img)
    draw.line(points, fill=(255, 0, 0), width=3)

    for i, (px, py) in enumerate(points[:-1]):
        radius = 6
        draw.ellipse([px-radius, py-radius, px+radius, py+radius],
                     fill=(0, 255, 0), outline=(0, 200, 0))

    if info_text:
        draw.rectangle([0, 0, w, 25], fill=(0, 0, 0))
        draw.text((5, 5), info_text, fill=(255, 255, 255))

    img.save(output_path)


def main():
    dataset_root = Path("/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new")
    output_dir = Path("/Volumes/ZX20/ML-Models/DocCornerNet-CoordClass/small_bbox_all_except_idcard_final")
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    area_threshold = 0.05
    dim_threshold = 0.1

    label_files = list(labels_dir.glob("*.txt"))
    print(f"Found {len(label_files)} label files")

    small_bbox_images = []

    for label_file in tqdm(label_files, desc="Scanning labels"):
        # Skip idcard_final
        if label_file.name.startswith("idcard_final"):
            continue

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

    small_bbox_images.sort(key=lambda x: x['area'])

    print(f"\nFound {len(small_bbox_images)} images with small bboxes (excluding idcard_final)")
    print("Generating visualizations...")

    for info in tqdm(small_bbox_images, desc="Drawing bboxes"):
        info_text = f"A:{info['area']*100:.1f}% W:{info['width']*100:.0f}% H:{info['height']*100:.0f}%"
        output_file = output_dir / info['image_name']
        draw_bbox_on_image(info['image_path'], info['coords'], str(output_file), info_text)

    print(f"\nAll {len(small_bbox_images)} images saved to: {output_dir}")

    # Save list
    list_file = output_dir / "image_list.txt"
    with open(list_file, 'w') as f:
        for info in small_bbox_images:
            f.write(f"{info['image_name']}\t{info['area']:.6f}\n")
    print(f"List saved to: {list_file}")


if __name__ == "__main__":
    main()
