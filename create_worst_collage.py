"""
Generate collage of worst 10 cases from validation set.
Shows GT (green) and prediction (red) bounding boxes.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm

import tensorflow as tf
from model import load_inference_model

# Constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_array, img_size):
    img = Image.fromarray(img_array).resize((img_size, img_size), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img


def load_gt(label_path):
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] == '0':  # GT class
                coords = list(map(float, parts[1:9]))
                return np.array(coords)
    return None


def compute_iou(pred, gt):
    """Compute IoU between two quadrilaterals."""
    try:
        from shapely.geometry import Polygon
        pred_poly = Polygon([(pred[i], pred[i+1]) for i in range(0, 8, 2)])
        gt_poly = Polygon([(gt[i], gt[i+1]) for i in range(0, 8, 2)])
        if not pred_poly.is_valid or not gt_poly.is_valid:
            return 0.0
        inter = pred_poly.intersection(gt_poly).area
        union = pred_poly.union(gt_poly).area
        return inter / union if union > 0 else 0.0
    except:
        return 0.0


def draw_quad(draw, coords, color, width, img_size):
    """Draw quadrilateral on image."""
    points = [(coords[i] * img_size, coords[i+1] * img_size) for i in range(0, 8, 2)]
    points.append(points[0])  # close polygon
    draw.line(points, fill=color, width=width)
    for p in points[:-1]:
        draw.ellipse([p[0]-4, p[1]-4, p[0]+4, p[1]+4], fill=color)


def main():
    # Load config and model
    checkpoint_path = Path('checkpoints/mobilenetv2_224_clean')
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

    # Load val split
    data_root = Path('/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new')
    with open(data_root / 'val.txt') as f:
        image_names = [l.strip() for l in f if l.strip() and not l.strip().startswith('negative_')]

    print(f'Found {len(image_names)} positive images')

    # Evaluate all images
    results = []
    batch_size = 32
    batch_imgs = []
    batch_names = []
    batch_gts = []

    for name in tqdm(image_names, desc="Processing"):
        img_path = data_root / 'images' / name
        label_path = data_root / 'labels' / (Path(name).stem + '.txt')

        if not img_path.exists() or not label_path.exists():
            continue

        gt = load_gt(label_path)
        if gt is None:
            continue

        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        preprocessed = preprocess(img_array, img_size)

        batch_imgs.append(preprocessed)
        batch_names.append(name)
        batch_gts.append(gt)

        if len(batch_imgs) >= batch_size:
            batch_input = np.stack(batch_imgs, axis=0)
            preds = model.predict(batch_input, verbose=0)

            for i, pred in enumerate(preds):
                coords = pred[:8]
                iou = compute_iou(coords, batch_gts[i])
                results.append({
                    'name': batch_names[i],
                    'iou': iou,
                    'pred': coords,
                    'gt': batch_gts[i],
                })

            batch_imgs = []
            batch_names = []
            batch_gts = []

    # Process remaining
    if batch_imgs:
        batch_input = np.stack(batch_imgs, axis=0)
        preds = model.predict(batch_input, verbose=0)
        for i, pred in enumerate(preds):
            coords = pred[:8]
            iou = compute_iou(coords, batch_gts[i])
            results.append({
                'name': batch_names[i],
                'iou': iou,
                'pred': coords,
                'gt': batch_gts[i],
            })

    # Sort by IoU (worst first)
    results.sort(key=lambda x: x['iou'])
    worst_10 = results[:10]

    print('\n=== Worst 10 cases ===')
    for i, r in enumerate(worst_10):
        print(f"{i+1}. {r['name']}: IoU={r['iou']:.4f}")

    # Create collage
    tile_size = 400
    cols = 5
    rows = 2
    collage = Image.new('RGB', (cols * tile_size, rows * tile_size), (40, 40, 40))

    for idx, r in enumerate(worst_10):
        row = idx // cols
        col = idx % cols

        img_path = data_root / 'images' / r['name']
        img = Image.open(img_path).convert('RGB')
        img = img.resize((tile_size, tile_size), Image.BILINEAR)

        draw = ImageDraw.Draw(img)

        # Draw GT (green)
        draw_quad(draw, r['gt'], (0, 255, 0), 3, tile_size)

        # Draw prediction (red)
        draw_quad(draw, r['pred'], (255, 0, 0), 3, tile_size)

        # Add IoU text
        draw.rectangle([0, 0, tile_size, 40], fill=(0, 0, 0, 180))
        draw.text((5, 5), f"IoU: {r['iou']:.3f}", fill=(255, 255, 0))
        draw.text((5, 22), r['name'][:40], fill=(255, 255, 255))

        collage.paste(img, (col * tile_size, row * tile_size))

    output_path = 'worst_10_val_cases.png'
    collage.save(output_path)
    print(f'\nSaved collage to {output_path}')
    print('Legend: GREEN = GT, RED = Prediction')


if __name__ == "__main__":
    main()
