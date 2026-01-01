"""
Generate full evaluation CSV for entire dataset.
Outputs CSV with: split, filename, iou, err_mean, err_max, score
Sorted by IoU descending (best first).
"""

import json
import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
import csv

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from model import load_inference_model
from metrics import compute_polygon_iou

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def load_single_image_fast(args):
    """Load a single image optimized for threading."""
    name, data_root, img_size = args

    img_path = data_root / 'images' / name
    label_path = data_root / 'labels' / (Path(name).stem + '.txt')

    if not img_path.exists():
        return None

    gt = None
    if label_path.exists():
        try:
            with open(label_path, 'r') as f:
                line = f.readline()
                if line:
                    parts = line.split()
                    if len(parts) >= 9 and parts[0] == '0':
                        gt = np.array([float(parts[i]) for i in range(1, 9)], dtype=np.float32)
        except:
            pass

    if gt is None:
        return None

    try:
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img = img.resize((img_size, img_size), Image.BILINEAR)
            return (name, np.asarray(img, dtype=np.uint8).copy(), gt)
    except:
        return None


def load_images_threaded(image_names, data_root, img_size, num_workers=64):
    """Load images using optimized threading."""
    args_list = [(name, data_root, img_size) for name in image_names]
    results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(load_single_image_fast, args_list),
                          total=len(args_list), desc="Loading images", unit="img"):
            if result is not None:
                results.append(result)

    return results


def compute_pixel_error(pred, gt, img_size):
    """Compute mean/max pixel error between pred and gt corners."""
    pred_px = pred.reshape(4, 2) * img_size
    gt_px = gt.reshape(4, 2) * img_size
    errors = np.linalg.norm(pred_px - gt_px, axis=1)
    return errors.mean(), errors.max()


def evaluate_split_gpu(model, data_root, split_name, img_size, batch_size=2048, num_workers=32):
    """Evaluate all images in a split."""
    split_file = data_root / f'{split_name}.txt'
    if not split_file.exists():
        print(f"Split file not found: {split_file}")
        return []

    with open(split_file) as f:
        image_names = [l.strip() for l in f if l.strip() and not l.strip().startswith('negative_')]

    print(f"\nEvaluating {split_name}: {len(image_names)} positive images")

    # Load images
    loaded = load_images_threaded(image_names, data_root, img_size, num_workers)
    n_samples = len(loaded)
    print(f"  Valid samples: {n_samples}")

    if n_samples == 0:
        return []

    # Prepare arrays with progress
    all_names = [None] * n_samples
    all_images = np.empty((n_samples, img_size, img_size, 3), dtype=np.uint8)
    all_gt = np.empty((n_samples, 8), dtype=np.float32)
    for i, (name, img, gt) in enumerate(
        tqdm(loaded, total=n_samples, desc="Stacking arrays", unit="img")
    ):
        all_names[i] = name
        all_images[i] = img
        all_gt[i] = gt
    del loaded

    # Normalize
    @tf.function
    def normalize_batch(images):
        x = tf.cast(images, tf.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x

    # Inference
    print(f"  Running inference (batch_size={batch_size})...")
    all_preds = []

    for i in tqdm(range(0, n_samples, batch_size), desc="Inference"):
        batch_imgs = all_images[i:i+batch_size]
        batch_normalized = normalize_batch(batch_imgs)
        preds = model(batch_normalized, training=False)
        if isinstance(preds, (list, tuple)):
            coords = preds[0].numpy()
        else:
            coords = preds.numpy()
        all_preds.append(coords)

    all_preds = np.concatenate(all_preds, axis=0)

    # Compute metrics
    print(f"  Computing metrics...")
    results = []
    for i in tqdm(range(n_samples), desc="Metrics"):
        pred = all_preds[i]
        gt = all_gt[i]
        name = all_names[i]

        iou = compute_polygon_iou(pred, gt)
        err_mean, err_max = compute_pixel_error(pred, gt, img_size)

        # Compute score (confidence-like metric)
        score = iou * 100 - err_mean

        results.append({
            'split': split_name,
            'name': name,
            'iou': iou,
            'err_mean': err_mean,
            'err_max': err_max,
            'score': score,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/workspace/checkpoints/mobilenetv2_256',
                        help='Checkpoint directory')
    parser.add_argument('--data_root', type=str, default='/workspace/doc-scanner-dataset',
                        help='Dataset root directory')
    parser.add_argument('--output', type=str, default='/workspace/full_evaluation.csv',
                        help='Output CSV file path')
    parser.add_argument('--splits', type=str, default='train,val',
                        help='Comma-separated list of splits to evaluate')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='Number of threads for image loading')
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("Full Dataset Evaluation to CSV")
    print("=" * 80)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs: {len(gpus)}")
        for gpu in gpus:
            print(f"  {gpu}")
    else:
        print("No GPU found, using CPU")

    # Load model
    checkpoint_dir = Path(args.checkpoint)
    data_root = Path(args.data_root)

    config_path = checkpoint_dir / 'config.json'
    with open(config_path) as f:
        config = json.load(f)

    img_size = config.get('img_size', 224)
    print(f"Image size: {img_size}")

    model = load_inference_model(
        checkpoint_dir / 'best_model.weights.h5',
        backbone=config.get('backbone', 'mobilenetv3_small'),
        alpha=config.get('alpha', 0.35),
        fpn_ch=config.get('fpn_ch', 32),
        simcc_ch=config.get('simcc_ch', 96),
        img_size=img_size,
        num_bins=config.get('num_bins', img_size),
    )
    print("Model loaded")

    # Warmup
    print("Warming up GPU...")
    dummy_input = tf.random.normal([64, img_size, img_size, 3])
    _ = model(dummy_input, training=False)
    del dummy_input
    print("Warmup done")

    # Evaluate all splits
    all_results = []
    splits = [s.strip() for s in args.splits.split(',')]

    for split in splits:
        results = evaluate_split_gpu(model, data_root, split, img_size,
                                      args.batch_size, args.num_workers)
        all_results.extend(results)

    if not all_results:
        print("No results found!")
        sys.exit(1)

    # Sort by IoU descending (best first)
    all_results.sort(key=lambda x: -x['iou'])

    # Write CSV
    output_path = Path(args.output)
    print(f"\nWriting {len(all_results)} results to {output_path}...")

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'filename', 'iou', 'err_mean', 'err_max', 'score'])
        for r in all_results:
            writer.writerow([
                r['split'],
                r['name'],
                f"{r['iou']:.6f}",
                f"{r['err_mean']:.2f}",
                f"{r['err_max']:.2f}",
                f"{r['score']:.4f}",
            ])

    # Print summary
    train_results = [r for r in all_results if r['split'] == 'train']
    val_results = [r for r in all_results if r['split'] == 'val']

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total images: {len(all_results)}")
    print(f"  Train: {len(train_results)}")
    print(f"  Val: {len(val_results)}")

    if all_results:
        ious = [r['iou'] for r in all_results]
        errs = [r['err_mean'] for r in all_results]
        print(f"\nIoU: mean={np.mean(ious):.4f}, min={np.min(ious):.4f}, max={np.max(ious):.4f}")
        print(f"Error: mean={np.mean(errs):.2f}px, min={np.min(errs):.2f}px, max={np.max(errs):.2f}px")

    print(f"\nOutput: {output_path}")
    print("DONE!")


if __name__ == "__main__":
    main()
