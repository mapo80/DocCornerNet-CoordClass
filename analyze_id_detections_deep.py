#!/usr/bin/env python3
"""
Deep analysis of id_detections dataset to find all potential issues:
1. Small bboxes (already done)
2. Extreme aspect ratios (already done)
3. Anomalous angles (already done)
4. Documents at image edges (cropped documents)
5. Very dark or bright images
6. Blurry images
7. Multiple documents or no document visible
8. Annotations that don't match visible content
9. Duplicate or near-duplicate images
10. Wrong corner ordering
11. Document occupies unusual position
12. Image resolution issues
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageStat, ImageFilter
from tqdm import tqdm
from collections import defaultdict
import hashlib


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


def check_corner_ordering(coords):
    """
    Check if corners are in correct order (TL, TR, BR, BL).
    Returns issues if ordering seems wrong.
    """
    # coords: [x0,y0, x1,y1, x2,y2, x3,y3] = TL, TR, BR, BL
    x = coords[0::2]
    y = coords[1::2]

    issues = []

    # TL should be top-left (small x, small y)
    # TR should be top-right (large x, small y)
    # BR should be bottom-right (large x, large y)
    # BL should be bottom-left (small x, large y)

    # Check if top corners have smaller y than bottom
    if y[0] > y[3] + 0.1:  # TL.y > BL.y significantly
        issues.append("TL_below_BL")
    if y[1] > y[2] + 0.1:  # TR.y > BR.y significantly
        issues.append("TR_below_BR")

    # Check if left corners have smaller x than right
    if x[0] > x[1] + 0.1:  # TL.x > TR.x significantly
        issues.append("TL_right_of_TR")
    if x[3] > x[2] + 0.1:  # BL.x > BR.x significantly
        issues.append("BL_right_of_BR")

    return issues


def check_edge_proximity(coords, threshold=0.02):
    """Check if document is too close to image edges (possibly cropped)."""
    x = coords[0::2]
    y = coords[1::2]

    issues = []

    # Check proximity to each edge
    if x.min() < threshold:
        issues.append(f"touches_left_edge:{x.min():.3f}")
    if x.max() > 1 - threshold:
        issues.append(f"touches_right_edge:{x.max():.3f}")
    if y.min() < threshold:
        issues.append(f"touches_top_edge:{y.min():.3f}")
    if y.max() > 1 - threshold:
        issues.append(f"touches_bottom_edge:{y.max():.3f}")

    return issues


def analyze_image_quality(image_path):
    """Analyze image for quality issues."""
    issues = []

    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        # 1. Resolution check
        if w < 200 or h < 200:
            issues.append(f"low_resolution:{w}x{h}")

        # 2. Brightness check
        stat = ImageStat.Stat(img)
        brightness = sum(stat.mean) / 3

        if brightness < 30:
            issues.append(f"very_dark:brightness={brightness:.1f}")
        elif brightness < 50:
            issues.append(f"dark:brightness={brightness:.1f}")
        elif brightness > 240:
            issues.append(f"very_bright:brightness={brightness:.1f}")
        elif brightness > 220:
            issues.append(f"bright:brightness={brightness:.1f}")

        # 3. Contrast check (std deviation)
        contrast = sum(stat.stddev) / 3
        if contrast < 20:
            issues.append(f"low_contrast:stddev={contrast:.1f}")

        # 4. Blur detection (Laplacian variance)
        gray = img.convert("L")
        laplacian = gray.filter(ImageFilter.FIND_EDGES)
        lap_stat = ImageStat.Stat(laplacian)
        sharpness = lap_stat.var[0]

        if sharpness < 100:
            issues.append(f"possibly_blurry:variance={sharpness:.1f}")

        # 5. Aspect ratio of image itself
        img_aspect = max(w/h, h/w)
        if img_aspect > 3:
            issues.append(f"unusual_image_aspect:{img_aspect:.1f}")

    except Exception as e:
        issues.append(f"error_reading_image:{str(e)[:50]}")

    return issues


def compute_image_hash(image_path, hash_size=8):
    """Compute perceptual hash for duplicate detection."""
    try:
        img = Image.open(image_path).convert("L")
        img = img.resize((hash_size + 1, hash_size), Image.LANCZOS)
        pixels = list(img.getdata())

        diff = []
        for row in range(hash_size):
            for col in range(hash_size):
                left = pixels[row * (hash_size + 1) + col]
                right = pixels[row * (hash_size + 1) + col + 1]
                diff.append(left > right)

        return ''.join(['1' if b else '0' for b in diff])
    except:
        return None


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
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]  # TL=red, TR=green, BR=blue, BL=yellow
        draw.ellipse([px-radius, py-radius, px+radius, py+radius],
                     fill=colors[i], outline=(255,255,255))
        draw.text((px+10, py-10), f"{i}", fill=(255, 255, 255))

    if info_text:
        lines = info_text.split('\n')
        y_offset = 5
        for line in lines[:4]:
            draw.rectangle([0, y_offset-2, w, y_offset+15], fill=(0, 0, 0))
            draw.text((5, y_offset), line[:120], fill=(255, 255, 255))
            y_offset += 18

    img.save(output_path)


def main():
    dataset_root = Path("/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new")
    output_dir = Path("/Volumes/ZX20/ML-Models/DocCornerNet-CoordClass/full_dataset_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    # Get ALL labels (not just id_detections)
    label_files = list(labels_dir.glob("*.txt"))
    print(f"Found {len(label_files)} labels")

    all_issues = defaultdict(list)
    image_hashes = defaultdict(list)  # hash -> list of image names

    for label_file in tqdm(label_files, desc="Analyzing"):
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

        issues = []

        # 1. Corner ordering check
        corner_issues = check_corner_ordering(coords)
        issues.extend(corner_issues)

        # 2. Edge proximity (cropped documents)
        edge_issues = check_edge_proximity(coords)
        issues.extend(edge_issues)

        # 3. Image quality checks
        quality_issues = analyze_image_quality(str(image_path))
        issues.extend(quality_issues)

        # 4. Compute hash for duplicate detection
        img_hash = compute_image_hash(str(image_path))
        if img_hash:
            image_hashes[img_hash].append(image_path.name)

        if issues:
            for issue in issues:
                issue_type = issue.split(':')[0]
                all_issues[issue_type].append({
                    'image_name': image_path.name,
                    'image_path': str(image_path),
                    'coords': coords,
                    'issues': issues
                })

    # Find duplicates
    duplicates = {h: imgs for h, imgs in image_hashes.items() if len(imgs) > 1}
    if duplicates:
        for hash_val, imgs in duplicates.items():
            for img_name in imgs:
                img_path = images_dir / img_name
                label_path = labels_dir / f"{Path(img_name).stem}.txt"
                coords, _ = load_yolo_obb_label(str(label_path))
                all_issues['duplicate_image'].append({
                    'image_name': img_name,
                    'image_path': str(img_path),
                    'coords': coords,
                    'issues': [f"duplicate_of:{','.join(imgs)}"]
                })

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY OF ISSUES FOUND IN FULL DATASET")
    print("="*70)

    for issue_type, images in sorted(all_issues.items(), key=lambda x: -len(x[1])):
        print(f"\n{issue_type}: {len(images)} images")
        for img in images[:3]:
            print(f"  - {img['image_name']}")
        if len(images) > 3:
            print(f"  ... and {len(images) - 3} more")

    # Save visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    for issue_type, images in all_issues.items():
        issue_dir = output_dir / issue_type
        issue_dir.mkdir(parents=True, exist_ok=True)

        # Save up to 50 examples
        for img in tqdm(images[:50], desc=f"Drawing {issue_type}", leave=False):
            info_text = '\n'.join(img['issues'][:4])
            output_file = issue_dir / img['image_name']
            try:
                draw_bbox_on_image(img['image_path'], img['coords'], str(output_file), info_text)
            except Exception as e:
                print(f"Error drawing {img['image_name']}: {e}")

        # Save full list
        list_file = issue_dir / "image_list.txt"
        with open(list_file, 'w') as f:
            for img in images:
                f.write(f"{img['image_name']}\t{'; '.join(img['issues'])}\n")

    print(f"\nResults saved to: {output_dir}")

    # Total unique problematic
    all_problematic = set()
    for images in all_issues.values():
        for img in images:
            all_problematic.add(img['image_name'])
    print(f"Total unique problematic images: {len(all_problematic)}")


if __name__ == "__main__":
    main()
