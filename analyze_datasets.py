#!/usr/bin/env python3
"""
Analyze id_detections and id_card datasets for potential issues:
- Small bboxes (already done)
- Very large bboxes (document fills entire image)
- Extreme aspect ratios
- Bbox outside image bounds
- Very skewed/distorted quadrilaterals
- Near-degenerate polygons (very thin or collapsed)
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from collections import defaultdict
import math


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
    """Shoelace formula for quadrilateral area."""
    x = coords[0::2]
    y = coords[1::2]
    n = len(x)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j]
        area -= x[j] * y[i]
    return abs(area) / 2.0


def compute_edge_lengths(coords):
    """Compute lengths of 4 edges."""
    lengths = []
    for i in range(4):
        j = (i + 1) % 4
        x1, y1 = coords[i*2], coords[i*2+1]
        x2, y2 = coords[j*2], coords[j*2+1]
        length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        lengths.append(length)
    return lengths


def compute_angles(coords):
    """Compute interior angles of quadrilateral."""
    angles = []
    for i in range(4):
        # Get three consecutive points
        p0 = i - 1 if i > 0 else 3
        p1 = i
        p2 = (i + 1) % 4

        x0, y0 = coords[p0*2], coords[p0*2+1]
        x1, y1 = coords[p1*2], coords[p1*2+1]
        x2, y2 = coords[p2*2], coords[p2*2+1]

        # Vectors
        v1 = (x0 - x1, y0 - y1)
        v2 = (x2 - x1, y2 - y1)

        # Angle between vectors
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if mag1 * mag2 < 1e-10:
            angles.append(0)
        else:
            cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
            angle = math.degrees(math.acos(cos_angle))
            angles.append(angle)

    return angles


def is_convex(coords):
    """Check if quadrilateral is convex."""
    def cross_product_sign(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    points = [(coords[i*2], coords[i*2+1]) for i in range(4)]
    signs = []
    for i in range(4):
        o = points[i]
        a = points[(i+1) % 4]
        b = points[(i+2) % 4]
        signs.append(cross_product_sign(o, a, b) > 0)

    return all(signs) or not any(signs)


def analyze_image(image_path, coords):
    """Analyze single image and return issues."""
    issues = []

    x = coords[0::2]
    y = coords[1::2]

    # 1. Coords outside [0, 1]
    if x.min() < -0.01 or x.max() > 1.01 or y.min() < -0.01 or y.max() > 1.01:
        issues.append(f"coords_outside_bounds: x=[{x.min():.3f},{x.max():.3f}] y=[{y.min():.3f},{y.max():.3f}]")

    # 2. Bbox dimensions
    width = x.max() - x.min()
    height = y.max() - y.min()
    area = compute_polygon_area(coords)

    # Very large bbox (> 95% of image)
    if area > 0.95:
        issues.append(f"very_large_bbox: area={area:.3f}")

    # Extreme aspect ratio
    if width > 0 and height > 0:
        aspect = max(width/height, height/width)
        if aspect > 5:
            issues.append(f"extreme_aspect_ratio: {aspect:.1f}")

    # 3. Edge lengths
    edges = compute_edge_lengths(coords)
    min_edge = min(edges)
    max_edge = max(edges)

    # Very short edge (degenerate)
    if min_edge < 0.02:
        issues.append(f"very_short_edge: {min_edge:.4f}")

    # Edge ratio (one edge much longer than another)
    if min_edge > 0:
        edge_ratio = max_edge / min_edge
        if edge_ratio > 10:
            issues.append(f"edge_ratio_extreme: {edge_ratio:.1f}")

    # 4. Angles
    angles = compute_angles(coords)
    min_angle = min(angles)
    max_angle = max(angles)

    # Very acute angle (< 20 degrees)
    if min_angle < 20:
        issues.append(f"very_acute_angle: {min_angle:.1f}°")

    # Very obtuse angle (> 160 degrees)
    if max_angle > 160:
        issues.append(f"very_obtuse_angle: {max_angle:.1f}°")

    # 5. Non-convex quadrilateral
    if not is_convex(coords):
        issues.append("non_convex_quad")

    # 6. Self-intersecting check (simplified)
    # Check if area is very small compared to bounding box
    bbox_area = width * height
    if bbox_area > 0.01 and area / bbox_area < 0.3:
        issues.append(f"possibly_self_intersecting: area_ratio={area/bbox_area:.2f}")

    return issues


def draw_bbox_on_image(image_path, coords, output_path, info_text=""):
    """Draw bbox and save image."""
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
        # Multi-line text
        lines = info_text.split('\n')
        y_offset = 5
        for line in lines[:3]:  # Max 3 lines
            draw.rectangle([0, y_offset-2, w, y_offset+15], fill=(0, 0, 0, 180))
            draw.text((5, y_offset), line[:100], fill=(255, 255, 255))
            y_offset += 18

    img.save(output_path)


def main():
    dataset_root = Path("/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new")
    output_dir = Path("/Volumes/ZX20/ML-Models/DocCornerNet-CoordClass/dataset_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    # Analyze only id_detections and id_card
    prefixes = ["id_detections", "id_card"]

    all_issues = defaultdict(list)

    for prefix in prefixes:
        print(f"\n=== Analyzing {prefix} ===")
        label_files = list(labels_dir.glob(f"{prefix}*.txt"))
        print(f"Found {len(label_files)} labels")

        for label_file in tqdm(label_files, desc=f"Analyzing {prefix}"):
            coords, has_doc = load_yolo_obb_label(str(label_file))
            if not has_doc:
                continue

            image_stem = label_file.stem
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                candidate = images_dir / f"{image_stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if not image_path:
                continue

            issues = analyze_image(str(image_path), coords)

            if issues:
                for issue in issues:
                    issue_type = issue.split(':')[0]
                    all_issues[issue_type].append({
                        'image_name': image_path.name,
                        'image_path': str(image_path),
                        'coords': coords,
                        'issues': issues
                    })

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF ISSUES FOUND")
    print("="*60)

    for issue_type, images in sorted(all_issues.items(), key=lambda x: -len(x[1])):
        print(f"\n{issue_type}: {len(images)} images")
        # Show first 5 examples
        for img in images[:5]:
            print(f"  - {img['image_name']}")

    # Save visualizations for each issue type
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    for issue_type, images in all_issues.items():
        issue_dir = output_dir / issue_type
        issue_dir.mkdir(parents=True, exist_ok=True)

        # Save up to 30 examples per issue type
        for img in tqdm(images[:30], desc=f"Drawing {issue_type}", leave=False):
            info_text = '\n'.join(img['issues'][:3])
            output_file = issue_dir / img['image_name']
            draw_bbox_on_image(img['image_path'], img['coords'], str(output_file), info_text)

        # Save full list
        list_file = issue_dir / "image_list.txt"
        with open(list_file, 'w') as f:
            for img in images:
                f.write(f"{img['image_name']}\t{'; '.join(img['issues'])}\n")

    print(f"\nAll results saved to: {output_dir}")

    # Print total unique problematic images
    all_problematic = set()
    for images in all_issues.values():
        for img in images:
            all_problematic.add(img['image_name'])

    print(f"\nTotal unique problematic images: {len(all_problematic)}")


if __name__ == "__main__":
    main()
