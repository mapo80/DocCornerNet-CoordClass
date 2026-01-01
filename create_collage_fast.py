"""
Fast GPU-accelerated collage generator for worst/best cases.
Uses batched inference for speed.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import tensorflow as tf
from model import load_inference_model

# Constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def compute_iou_batch(pred_coords, gt_coords):
    """
    Compute IoU for batch of quadrilaterals using vectorized operations.
    Approximates IoU using bounding box for speed.
    """
    ious = []
    for pred, gt in zip(pred_coords, gt_coords):
        # Get bounding boxes
        pred_x = pred[0::2]
        pred_y = pred[1::2]
        gt_x = gt[0::2]
        gt_y = gt[1::2]

        # Intersection
        x1 = max(min(pred_x), min(gt_x))
        y1 = max(min(pred_y), min(gt_y))
        x2 = min(max(pred_x), max(gt_x))
        y2 = min(max(pred_y), max(gt_y))

        if x2 <= x1 or y2 <= y1:
            ious.append(0.0)
            continue

        inter = (x2 - x1) * (y2 - y1)

        # Areas (approximate with bounding box)
        pred_area = (max(pred_x) - min(pred_x)) * (max(pred_y) - min(pred_y))
        gt_area = (max(gt_x) - min(gt_x)) * (max(gt_y) - min(gt_y))

        union = pred_area + gt_area - inter
        ious.append(inter / union if union > 0 else 0.0)

    return np.array(ious)


def draw_quad(draw, coords, color, width, size):
    """Draw quadrilateral on image."""
    points = [(coords[i] * size, coords[i+1] * size) for i in range(0, 8, 2)]
    points.append(points[0])
    draw.line(points, fill=color, width=width)
    for p in points[:-1]:
        r = 5
        draw.ellipse([p[0]-r, p[1]-r, p[0]+r, p[1]+r], fill=color)


def main():
    parser = argparse.ArgumentParser(description="Generate worst/best cases collage")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root")
    parser.add_argument("--split", type=str, default="val", help="Split to evaluate")
    parser.add_argument("--num_cases", type=int, default=10, help="Number of worst/best cases")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--output", type=str, default="collage.png", help="Output filename")
    parser.add_argument("--mode", type=str, choices=["worst", "best", "both"], default="both")
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

    # Preload all data
    print("Loading images and labels...")
    all_images = []
    all_names = []
    all_gts = []

    for name in tqdm(image_names, desc="Loading"):
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
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((img_size, img_size), Image.BILINEAR)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD

        all_images.append(img_array)
        all_names.append(name)
        all_gts.append(gt)

    print(f"Loaded {len(all_images)} valid images")

    # Batch inference
    print("Running inference...")
    all_preds = []
    all_scores = []

    for i in tqdm(range(0, len(all_images), args.batch_size), desc="Inference"):
        batch = np.stack(all_images[i:i+args.batch_size], axis=0)
        preds = model.predict(batch, verbose=0)
        all_preds.extend(preds[:, :8])
        all_scores.extend(preds[:, 8])

    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)

    # Compute IoUs
    print("Computing IoUs...")
    ious = compute_iou_batch(all_preds, all_gts)

    # Sort by IoU
    sorted_indices = np.argsort(ious)
    worst_indices = sorted_indices[:args.num_cases]
    best_indices = sorted_indices[-args.num_cases:][::-1]

    # Create collages
    tile_size = 400
    cols = 5

    def create_collage(indices, title):
        rows = (len(indices) + cols - 1) // cols
        collage = Image.new('RGB', (cols * tile_size, rows * tile_size + 40), (40, 40, 40))
        draw_collage = ImageDraw.Draw(collage)

        # Title
        draw_collage.text((10, 10), title, fill=(255, 255, 255))

        for idx, data_idx in enumerate(indices):
            row = idx // cols
            col = idx % cols

            # Reload original image for display
            name = all_names[data_idx]
            img_path = data_root / 'images' / name
            img = Image.open(img_path).convert('RGB')
            img = img.resize((tile_size, tile_size), Image.BILINEAR)

            draw = ImageDraw.Draw(img)

            # Draw GT (green)
            draw_quad(draw, all_gts[data_idx], (0, 255, 0), 3, tile_size)

            # Draw prediction (red)
            draw_quad(draw, all_preds[data_idx], (255, 0, 0), 3, tile_size)

            # Add text overlay
            draw.rectangle([0, 0, tile_size, 50], fill=(0, 0, 0, 200))
            draw.text((5, 5), f"IoU: {ious[data_idx]:.3f}", fill=(255, 255, 0))
            draw.text((5, 25), name[:45], fill=(255, 255, 255))

            collage.paste(img, (col * tile_size, row * tile_size + 40))

        return collage

    if args.mode == "worst":
        collage = create_collage(worst_indices, f"WORST {args.num_cases} cases (GREEN=GT, RED=Pred)")
        output_name = args.output.replace('.png', '_worst.png')
        collage.save(output_name)
        print(f"Saved: {output_name}")

    elif args.mode == "best":
        collage = create_collage(best_indices, f"BEST {args.num_cases} cases (GREEN=GT, RED=Pred)")
        output_name = args.output.replace('.png', '_best.png')
        collage.save(output_name)
        print(f"Saved: {output_name}")

    else:  # both
        # Worst
        worst_collage = create_collage(worst_indices, f"WORST {args.num_cases} cases (GREEN=GT, RED=Pred)")
        worst_name = args.output.replace('.png', '_worst.png')
        worst_collage.save(worst_name)
        print(f"Saved: {worst_name}")

        # Best
        best_collage = create_collage(best_indices, f"BEST {args.num_cases} cases (GREEN=GT, RED=Pred)")
        best_name = args.output.replace('.png', '_best.png')
        best_collage.save(best_name)
        print(f"Saved: {best_name}")

    # Print stats
    print(f"\n=== Statistics ===")
    print(f"Mean IoU: {np.mean(ious):.4f}")
    print(f"Median IoU: {np.median(ious):.4f}")
    print(f"\nWorst {args.num_cases}:")
    for i, idx in enumerate(worst_indices):
        print(f"  {i+1}. {all_names[idx]}: IoU={ious[idx]:.4f}")
    print(f"\nBest {args.num_cases}:")
    for i, idx in enumerate(best_indices):
        print(f"  {i+1}. {all_names[idx]}: IoU={ious[idx]:.4f}")


if __name__ == "__main__":
    main()
