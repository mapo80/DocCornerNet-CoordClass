"""
Streaming collage generator - processes images one at a time.
Does NOT load all images to memory. Uses heap to track best/worst cases.
"""

import argparse
import json
import heapq
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm

import tensorflow as tf
from model import load_inference_model

# Constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def compute_iou_bbox(pred, gt):
    """Fast IoU using bounding box approximation."""
    pred_x = pred[0::2]
    pred_y = pred[1::2]
    gt_x = gt[0::2]
    gt_y = gt[1::2]

    # Intersection
    x1 = max(np.min(pred_x), np.min(gt_x))
    y1 = max(np.min(pred_y), np.min(gt_y))
    x2 = min(np.max(pred_x), np.max(gt_x))
    y2 = min(np.max(pred_y), np.max(gt_y))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    pred_area = (np.max(pred_x) - np.min(pred_x)) * (np.max(pred_y) - np.min(pred_y))
    gt_area = (np.max(gt_x) - np.min(gt_x)) * (np.max(gt_y) - np.min(gt_y))
    union = pred_area + gt_area - inter

    return inter / union if union > 0 else 0.0


def draw_quad(draw, coords, color, width, size):
    """Draw quadrilateral on image."""
    points = [(coords[i] * size, coords[i+1] * size) for i in range(0, 8, 2)]
    points.append(points[0])
    draw.line(points, fill=color, width=width)
    for p in points[:-1]:
        r = 4
        draw.ellipse([p[0]-r, p[1]-r, p[0]+r, p[1]+r], fill=color)


def main():
    parser = argparse.ArgumentParser(description="Streaming collage generator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root")
    parser.add_argument("--split", type=str, default="val", help="Split to evaluate")
    parser.add_argument("--num_cases", type=int, default=10, help="Number of worst/best cases")
    parser.add_argument("--output", type=str, default="collage.png", help="Output filename")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    data_root = Path(args.data_root)

    # Load config and model
    with open(checkpoint_path / 'config.json') as f:
        config = json.load(f)

    img_size = config.get('img_size', 224)
    print(f"Loading model (img_size={img_size})...")

    model = load_inference_model(
        checkpoint_path / 'best_model.weights.h5',
        backbone=config.get('backbone', 'mobilenetv2'),
        alpha=config.get('alpha', 0.35),
        fpn_ch=config.get('fpn_ch', 32),
        simcc_ch=config.get('simcc_ch', 96),
        img_size=img_size,
        num_bins=config.get('num_bins', img_size),
    )
    print('Model loaded')

    # Load split file
    split_file = data_root / f'{args.split}.txt'
    if not split_file.exists():
        split_file = data_root / f'{args.split}_with_negative_v2.txt'

    with open(split_file) as f:
        image_names = [l.strip() for l in f if l.strip() and not l.strip().startswith('negative_')]

    print(f'Found {len(image_names)} positive images')

    # Heaps for tracking best/worst cases
    worst_heap = []  # min-heap: stores (-iou, name, pred, gt)
    best_heap = []   # min-heap: stores (iou, name, pred, gt)

    all_ious = []

    # Process in batches (but don't preload all images)
    print(f"Processing images (batch_size={args.batch_size})...")

    batch_imgs = []
    batch_names = []
    batch_gts = []

    def process_batch():
        """Process accumulated batch and update heaps."""
        if not batch_imgs:
            return

        batch_input = np.stack(batch_imgs, axis=0)
        pred = model.predict(batch_input, verbose=0)
        coords_batch = pred[0]  # shape (batch, 8)

        for i in range(len(batch_imgs)):
            coords = coords_batch[i]
            gt = batch_gts[i]
            name = batch_names[i]

            iou = compute_iou_bbox(coords, gt)
            all_ious.append(iou)

            # Update worst heap (keep lowest IoUs)
            if len(worst_heap) < args.num_cases:
                heapq.heappush(worst_heap, (-iou, name, coords.copy(), gt.copy()))
            elif iou < -worst_heap[0][0]:
                heapq.heapreplace(worst_heap, (-iou, name, coords.copy(), gt.copy()))

            # Update best heap (keep highest IoUs)
            if len(best_heap) < args.num_cases:
                heapq.heappush(best_heap, (iou, name, coords.copy(), gt.copy()))
            elif iou > best_heap[0][0]:
                heapq.heapreplace(best_heap, (iou, name, coords.copy(), gt.copy()))

    for name in tqdm(image_names, desc="Inference"):
        img_path = data_root / 'images' / name
        label_path = data_root / 'labels' / (Path(name).stem + '.txt')

        if not img_path.exists() or not label_path.exists():
            continue

        # Load GT
        gt = None
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts[0] == '0':
                    gt = np.array(list(map(float, parts[1:9])))
                    break
        if gt is None:
            continue

        # Load and preprocess image
        try:
            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize((img_size, img_size), Image.BILINEAR)
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD
        except Exception as e:
            continue

        batch_imgs.append(img_array)
        batch_names.append(name)
        batch_gts.append(gt)

        # Process batch when full
        if len(batch_imgs) >= args.batch_size:
            process_batch()
            batch_imgs = []
            batch_names = []
            batch_gts = []

    # Process remaining
    process_batch()

    # Sort results
    worst_cases = sorted([(-h[0], h[1], h[2], h[3]) for h in worst_heap], key=lambda x: x[0])
    best_cases = sorted([h for h in best_heap], key=lambda x: -x[0])

    # Print stats
    all_ious = np.array(all_ious)
    print(f"\n=== Statistics ===")
    print(f"Mean IoU: {np.mean(all_ious):.4f}")
    print(f"Median IoU: {np.median(all_ious):.4f}")
    print(f"Min IoU: {np.min(all_ious):.4f}")
    print(f"Max IoU: {np.max(all_ious):.4f}")

    print(f"\nWorst {args.num_cases}:")
    for i, (iou, name, _, _) in enumerate(worst_cases):
        print(f"  {i+1}. {name}: IoU={iou:.4f}")

    print(f"\nBest {args.num_cases}:")
    for i, (iou, name, _, _) in enumerate(best_cases):
        print(f"  {i+1}. {name}: IoU={iou:.4f}")

    # Now create collages - only load the 20 images we need
    tile_size = 400
    cols = 5

    def create_collage(cases, title):
        rows = (len(cases) + cols - 1) // cols
        collage = Image.new('RGB', (cols * tile_size, rows * tile_size + 40), (40, 40, 40))
        draw_collage = ImageDraw.Draw(collage)
        draw_collage.text((10, 10), title, fill=(255, 255, 255))

        for idx, (iou, name, pred, gt) in enumerate(cases):
            row = idx // cols
            col = idx % cols

            # Load this specific image
            img_path = data_root / 'images' / name
            img = Image.open(img_path).convert('RGB')
            img = img.resize((tile_size, tile_size), Image.BILINEAR)

            draw = ImageDraw.Draw(img)

            # Draw GT (green) and prediction (red)
            draw_quad(draw, gt, (0, 255, 0), 3, tile_size)
            draw_quad(draw, pred, (255, 0, 0), 3, tile_size)

            # Text overlay
            draw.rectangle([0, 0, tile_size, 50], fill=(0, 0, 0))
            draw.text((5, 5), f"IoU: {iou:.3f}", fill=(255, 255, 0))
            draw.text((5, 25), name[:45], fill=(255, 255, 255))

            collage.paste(img, (col * tile_size, row * tile_size + 40))

        return collage

    # Generate both collages
    print("\nGenerating collages...")

    worst_collage = create_collage(worst_cases, f"WORST {args.num_cases} cases (GREEN=GT, RED=Pred)")
    worst_name = args.output.replace('.png', '_worst.png')
    worst_collage.save(worst_name)
    print(f"Saved: {worst_name}")

    best_collage = create_collage(best_cases, f"BEST {args.num_cases} cases (GREEN=GT, RED=Pred)")
    best_name = args.output.replace('.png', '_best.png')
    best_collage.save(best_name)
    print(f"Saved: {best_name}")


if __name__ == "__main__":
    main()
