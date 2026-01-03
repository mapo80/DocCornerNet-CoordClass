#!/usr/bin/env python3
"""
Evaluate a teacher document detector on a dataset.
Compare predictions with ground truth bounding boxes and visualize worst cases.
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import json
from tqdm import tqdm
import shutil

def _load_detector(detector_root: str, backbone: str, head: str, verbose: bool):
    detector_root_path = Path(detector_root).resolve()
    if str(detector_root_path) not in sys.path:
        sys.path.insert(0, str(detector_root_path))
    try:
        from document_detector import DocumentDetector  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import DocumentDetector from {detector_root_path}/document_detector.py"
        ) from exc

    kwargs = {"verbose": bool(verbose)}
    if backbone:
        kwargs["backbone_path"] = backbone
    if head:
        kwargs["head_path"] = head
    return DocumentDetector(**kwargs)


def parse_yolo_label(label_path, img_width, img_height):
    """
    Parse YOLO format label file with 4 corner points.
    Format: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized coordinates)
    Returns corners in order: TL, TR, BR, BL (clockwise from top-left)
    """
    with open(label_path, 'r') as f:
        line = f.readline().strip()

    if not line:
        return None

    parts = line.split()
    if len(parts) < 9:
        return None

    # Parse normalized coordinates
    coords = [float(x) for x in parts[1:9]]

    # Convert to absolute coordinates
    corners = []
    for i in range(4):
        x = coords[i*2] * img_width
        y = coords[i*2 + 1] * img_height
        corners.append((x, y))

    # Sort corners clockwise from top-left
    return sort_corners_clockwise(corners)


def sort_corners_clockwise(corners):
    """Sort 4 corners clockwise starting from top-left."""
    if len(corners) != 4:
        return corners

    # Sort by y coordinate first (separate top from bottom)
    pts_sorted = sorted(corners, key=lambda p: p[1])

    # Top two (smallest y values)
    top = pts_sorted[:2]
    bottom = pts_sorted[2:]

    # Sort top by x (left first): TL, TR
    top = sorted(top, key=lambda p: p[0])

    # Sort bottom by x descending (right first for clockwise): BR, BL
    bottom = sorted(bottom, key=lambda p: p[0], reverse=True)

    # Clockwise order: TL, TR, BR, BL
    return [top[0], top[1], bottom[0], bottom[1]]


def compute_iou(corners1, corners2):
    """
    Compute IoU between two quadrilaterals.
    Uses shapely for accurate polygon intersection.
    """
    try:
        from shapely.geometry import Polygon
        from shapely.validation import make_valid

        poly1 = Polygon(corners1)
        poly2 = Polygon(corners2)

        if not poly1.is_valid:
            poly1 = make_valid(poly1)
        if not poly2.is_valid:
            poly2 = make_valid(poly2)

        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area

        if union == 0:
            return 0.0

        return intersection / union
    except Exception as e:
        # Fallback: compute bounding box IoU
        return compute_bbox_iou(corners1, corners2)


def compute_bbox_iou(corners1, corners2):
    """Compute IoU using bounding boxes of the corners."""
    def get_bbox(corners):
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return min(xs), min(ys), max(xs), max(ys)

    box1 = get_bbox(corners1)
    box2 = get_bbox(corners2)

    # Intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def compute_corner_distances(gt_corners, pred_corners, img_width, img_height):
    """
    Compute distance between GT and predicted corners.
    Returns normalized distances (as fraction of image diagonal).
    """
    diagonal = np.sqrt(img_width**2 + img_height**2)

    distances = []
    for gt, pred in zip(gt_corners, pred_corners):
        dist = np.sqrt((gt[0] - pred[0])**2 + (gt[1] - pred[1])**2)
        distances.append(dist / diagonal)

    return distances


def classify_error(iou, corner_distances, confidence):
    """
    Classify the type of error based on metrics.
    Returns a category string.
    """
    max_corner_dist = max(corner_distances) if corner_distances else 1.0
    avg_corner_dist = np.mean(corner_distances) if corner_distances else 1.0

    if iou >= 0.9:
        return "excellent"
    elif iou >= 0.7:
        return "good"
    elif iou >= 0.5:
        if max_corner_dist > 0.15:
            return "corner_outlier"
        else:
            return "moderate"
    elif iou >= 0.3:
        return "poor"
    elif iou > 0:
        return "very_poor"
    else:
        return "no_overlap"


def create_comparison_image(image_path, gt_corners, pred_corners, result, metrics, output_path):
    """Create visualization comparing GT and predicted corners."""
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    # Draw GT polygon (green)
    if gt_corners:
        for i in range(4):
            p1 = gt_corners[i]
            p2 = gt_corners[(i + 1) % 4]
            draw.line([p1, p2], fill=(0, 255, 0), width=3)

        # Draw GT corner markers
        for i, (x, y) in enumerate(gt_corners):
            r = 6
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(0, 255, 0), outline=(255, 255, 255))

    # Draw predicted polygon (red)
    if pred_corners:
        for i in range(4):
            p1 = pred_corners[i]
            p2 = pred_corners[(i + 1) % 4]
            draw.line([p1, p2], fill=(255, 0, 0), width=3)

        # Draw predicted corner markers
        for i, (x, y) in enumerate(pred_corners):
            r = 6
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(255, 0, 0), outline=(255, 255, 255))

    # Add info text
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()

    info_lines = [
        f"IoU: {metrics.get('iou', 0):.3f}",
        f"Conf: {result.get('confidence', 0):.3f}",
        f"Category: {metrics.get('category', 'N/A')}",
        f"Max corner dist: {metrics.get('max_corner_dist', 0)*100:.1f}%",
    ]

    # Draw text background
    text_height = 25 * len(info_lines)
    draw.rectangle([0, 0, 250, text_height + 10], fill=(0, 0, 0, 180))

    for i, line in enumerate(info_lines):
        draw.text((10, 10 + i * 25), line, fill=(255, 255, 255), font=font)

    # Legend
    legend_y = image.height - 50
    draw.rectangle([0, legend_y, 200, image.height], fill=(0, 0, 0, 180))
    draw.text((10, legend_y + 5), "Green: Ground Truth", fill=(0, 255, 0), font=font)
    draw.text((10, legend_y + 25), "Red: Prediction", fill=(255, 0, 0), font=font)

    image.save(output_path)


def evaluate_dataset(
    dataset_path,
    output_dir,
    detector_root: str,
    backbone: str,
    head: str,
    verbose: bool,
    num_worst=50,
    skip_negative=True,
):
    """
    Evaluate a teacher detector on the dataset.
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return

    if not labels_dir.exists():
        print(f"Error: Labels directory not found: {labels_dir}")
        return

    # Initialize detector
    print("Loading teacher model...")
    detector = _load_detector(detector_root=detector_root, backbone=backbone, head=head, verbose=verbose)

    # Get all images
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"Found {len(image_files)} images")

    results = []
    category_counts = defaultdict(int)

    for img_path in tqdm(image_files, desc="Evaluating"):
        # Get corresponding label
        label_path = labels_dir / (img_path.stem + ".txt")

        if not label_path.exists():
            if skip_negative:
                continue
            # Treat as negative sample (no document)
            gt_corners = None
        else:
            # Load image to get dimensions
            img = Image.open(img_path)
            img_width, img_height = img.size

            gt_corners = parse_yolo_label(label_path, img_width, img_height)

            if gt_corners is None:
                if skip_negative:
                    continue

        # Run detection
        try:
            result = detector.detect(str(img_path))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

        pred_corners = result.get('corners')

        # Compute metrics
        metrics = {}

        if gt_corners is not None and pred_corners is not None:
            img = Image.open(img_path)
            img_width, img_height = img.size

            iou = compute_iou(gt_corners, pred_corners)
            corner_distances = compute_corner_distances(gt_corners, pred_corners, img_width, img_height)
            category = classify_error(iou, corner_distances, result.get('confidence', 0))

            metrics = {
                'iou': iou,
                'corner_distances': corner_distances,
                'max_corner_dist': max(corner_distances),
                'avg_corner_dist': np.mean(corner_distances),
                'category': category,
            }
        elif gt_corners is None and pred_corners is None:
            metrics = {
                'iou': 1.0,  # True negative
                'category': 'true_negative',
                'corner_distances': [],
                'max_corner_dist': 0,
                'avg_corner_dist': 0,
            }
        elif gt_corners is None:
            metrics = {
                'iou': 0.0,  # False positive
                'category': 'false_positive',
                'corner_distances': [],
                'max_corner_dist': 0,
                'avg_corner_dist': 0,
            }
        else:
            metrics = {
                'iou': 0.0,  # False negative
                'category': 'false_negative',
                'corner_distances': [],
                'max_corner_dist': 0,
                'avg_corner_dist': 0,
            }

        category_counts[metrics['category']] += 1

        results.append({
            'image_path': str(img_path),
            'gt_corners': gt_corners,
            'pred_corners': pred_corners,
            'result': result,
            'metrics': metrics,
        })

    print(f"\nProcessed {len(results)} images")

    # Statistics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    ious = [r['metrics']['iou'] for r in results if r['metrics'].get('iou') is not None]
    if ious:
        print(f"\nIoU Statistics:")
        print(f"  Mean IoU: {np.mean(ious):.4f}")
        print(f"  Median IoU: {np.median(ious):.4f}")
        print(f"  Min IoU: {np.min(ious):.4f}")
        print(f"  Max IoU: {np.max(ious):.4f}")
        print(f"  Std IoU: {np.std(ious):.4f}")

        # IoU thresholds
        for thresh in [0.5, 0.7, 0.9]:
            above = sum(1 for iou in ious if iou >= thresh)
            print(f"  IoU >= {thresh}: {above}/{len(ious)} ({100*above/len(ious):.1f}%)")

    print(f"\nCategory Distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(results)
        print(f"  {cat}: {count} ({pct:.1f}%)")

    # Sort by IoU to find worst cases
    results_sorted = sorted(results, key=lambda x: x['metrics'].get('iou', 0))

    # Create output directories by category
    for cat in set(r['metrics']['category'] for r in results):
        cat_dir = output_dir / cat
        cat_dir.mkdir(exist_ok=True)

    # Save worst cases
    worst_dir = output_dir / "worst_cases"
    worst_dir.mkdir(exist_ok=True)

    print(f"\nSaving {num_worst} worst cases to {worst_dir}...")

    for i, r in enumerate(results_sorted[:num_worst]):
        img_path = Path(r['image_path'])
        output_path = worst_dir / f"{i:03d}_iou{r['metrics']['iou']:.3f}_{img_path.name}"

        create_comparison_image(
            r['image_path'],
            r['gt_corners'],
            r['pred_corners'],
            r['result'],
            r['metrics'],
            output_path
        )

    # Save per-category examples
    print(f"\nSaving examples by category...")

    for cat in category_counts:
        cat_results = [r for r in results if r['metrics']['category'] == cat]
        cat_dir = output_dir / cat

        # Save up to 20 examples per category
        for i, r in enumerate(cat_results[:20]):
            img_path = Path(r['image_path'])
            output_path = cat_dir / f"{i:03d}_iou{r['metrics']['iou']:.3f}_{img_path.name}"

            create_comparison_image(
                r['image_path'],
                r['gt_corners'],
                r['pred_corners'],
                r['result'],
                r['metrics'],
                output_path
            )

    # Save full results as JSON
    json_results = []
    for r in results:
        json_results.append({
            'image_path': r['image_path'],
            'iou': r['metrics']['iou'],
            'category': r['metrics']['category'],
            'confidence': r['result'].get('confidence', 0),
            'max_corner_dist': r['metrics'].get('max_corner_dist', 0),
            'avg_corner_dist': r['metrics'].get('avg_corner_dist', 0),
        })

    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump({
            'summary': {
                'total_images': len(results),
                'mean_iou': float(np.mean(ious)) if ious else 0,
                'median_iou': float(np.median(ious)) if ious else 0,
                'category_counts': dict(category_counts),
            },
            'results': json_results,
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print(f"  - evaluation_results.json: Full results")
    print(f"  - worst_cases/: {num_worst} worst predictions")
    for cat in category_counts:
        print(f"  - {cat}/: Up to 20 examples")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate teacher model on dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./teacher_evaluation",
        help="Output directory",
    )
    parser.add_argument(
        "--detector_root",
        type=str,
        default="vendor/teacher-detector",
        help="Path containing document_detector.py",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="",
        help="Optional teacher backbone .tflite (passed to DocumentDetector)",
    )
    parser.add_argument(
        "--head",
        type=str,
        default="",
        help="Optional teacher head .tflite (passed to DocumentDetector)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable detector verbose mode")
    parser.add_argument(
        "--num-worst",
        type=int,
        default=50,
        help="Number of worst cases to visualize",
    )
    parser.add_argument(
        "--include-negative",
        action="store_true",
        help="Include negative samples (images without documents)",
    )

    args = parser.parse_args()

    evaluate_dataset(
        args.dataset,
        args.output,
        detector_root=args.detector_root,
        backbone=args.backbone,
        head=args.head,
        verbose=args.verbose,
        num_worst=args.num_worst,
        skip_negative=not args.include_negative
    )


if __name__ == "__main__":
    main()
