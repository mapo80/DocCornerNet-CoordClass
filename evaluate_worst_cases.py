"""
Evaluate worst cases from train and val sets.
GPU-optimized version: loads all data to GPU memory, uses XLA + mixed precision.
Creates collage with GT (green) and prediction (red) bboxes.
"""

import json
import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing as mp

# Enable XLA before importing TensorFlow
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'

import tensorflow as tf

# Enable memory growth to avoid OOM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

from model import load_inference_model

# Constants
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


# Thread-local storage for reusing objects
_thread_local = threading.local()


def load_single_image_fast(args):
    """Load a single image optimized for threading."""
    name, data_root, img_size = args

    img_path = data_root / 'images' / name
    label_path = data_root / 'labels' / (Path(name).stem + '.txt')

    if not img_path.exists():
        return None

    # Load GT - optimized parsing
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
        # Use fastest resize method
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


def compute_iou_batch(pred_coords, gt_coords):
    """Compute IoU for batch using bounding box approximation (fast)."""
    # Reshape to [B, 4, 2]
    pred = pred_coords.reshape(-1, 4, 2)
    gt = gt_coords.reshape(-1, 4, 2)

    # Bounding box IoU (approximation, but fast)
    pred_min = pred.min(axis=1)
    pred_max = pred.max(axis=1)
    gt_min = gt.min(axis=1)
    gt_max = gt.max(axis=1)

    inter_min = np.maximum(pred_min, gt_min)
    inter_max = np.minimum(pred_max, gt_max)
    inter_wh = np.maximum(inter_max - inter_min, 0)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]

    pred_area = (pred_max - pred_min).prod(axis=1)
    gt_area = (gt_max - gt_min).prod(axis=1)
    union_area = pred_area + gt_area - inter_area + 1e-9

    return inter_area / union_area


def compute_iou_polygon(pred, gt):
    """Compute exact polygon IoU using Shapely (for final results)."""
    try:
        from shapely.geometry import Polygon
        from shapely.validation import make_valid

        # Create polygons from corner points
        # Format: [x0,y0,x1,y1,x2,y2,x3,y3] -> [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]
        pred_points = [(pred[i], pred[i+1]) for i in range(0, 8, 2)]
        gt_points = [(gt[i], gt[i+1]) for i in range(0, 8, 2)]

        pred_poly = Polygon(pred_points)
        gt_poly = Polygon(gt_points)

        # Fix invalid polygons (self-intersecting, etc.)
        if not pred_poly.is_valid:
            pred_poly = make_valid(pred_poly)
        if not gt_poly.is_valid:
            gt_poly = make_valid(gt_poly)

        # Handle case where make_valid returns a different geometry type
        if pred_poly.is_empty or gt_poly.is_empty:
            return 0.0
        if pred_poly.geom_type != 'Polygon' or gt_poly.geom_type != 'Polygon':
            # Fallback to bbox IoU if polygons are invalid
            return float(compute_iou_batch(pred.reshape(1, -1), gt.reshape(1, -1))[0])

        inter = pred_poly.intersection(gt_poly).area
        union = pred_poly.union(gt_poly).area
        return inter / union if union > 0 else 0.0
    except Exception as e:
        # Fallback to bbox IoU
        return float(compute_iou_batch(pred.reshape(1, -1), gt.reshape(1, -1))[0])


def compute_pixel_error(pred, gt, img_size):
    """Compute mean/max pixel error between pred and gt corners."""
    pred_px = pred.reshape(4, 2) * img_size
    gt_px = gt.reshape(4, 2) * img_size
    errors = np.linalg.norm(pred_px - gt_px, axis=1)
    return errors.mean(), errors.max()


def draw_quad(draw, coords, color, width, size):
    """Draw quadrilateral on image."""
    points = [(coords[i] * size, coords[i+1] * size) for i in range(0, 8, 2)]
    points.append(points[0])
    draw.line(points, fill=color, width=width)
    for p in points[:-1]:
        r = 5
        draw.ellipse([p[0]-r, p[1]-r, p[0]+r, p[1]+r], fill=color)


def evaluate_split_gpu(model, data_root, split_name, img_size, batch_size=2048, num_workers=32):
    """
    Evaluate all images using GPU-optimized pipeline.

    1. Load all images in parallel (multiprocessing, bypasses GIL)
    2. Stack into numpy arrays
    3. Run inference on GPU with large batches
    """
    split_file = data_root / f'{split_name}.txt'
    if not split_file.exists():
        print(f"Split file not found: {split_file}")
        return []

    with open(split_file) as f:
        image_names = [l.strip() for l in f if l.strip() and not l.strip().startswith('negative_')]

    print(f"\nEvaluating {split_name}: {len(image_names)} positive images")
    print(f"  batch_size={batch_size}, workers={num_workers}")

    # Step 1: Load all images in parallel (threading)
    print(f"  Loading images to RAM (threading)...")
    loaded = load_images_threaded(image_names, data_root, img_size, num_workers)

    n_samples = len(loaded)
    print(f"  Valid samples: {n_samples}")

    if n_samples == 0:
        return []

    # Step 2: Stack into numpy arrays (unpack tuples)
    print(f"  Preparing tensors...")
    all_names = [item[0] for item in loaded]
    all_images = np.stack([item[1] for item in loaded], axis=0)  # [N, H, W, 3] uint8
    all_gt = np.stack([item[2] for item in loaded], axis=0)  # [N, 8]
    del loaded  # Free memory

    mem_gb = all_images.nbytes / 1e9
    print(f"  Memory: {mem_gb:.2f} GB")

    # Step 3: Create TF dataset
    @tf.function
    def normalize_batch(images):
        """Normalize uint8 images to float32 with ImageNet stats."""
        images = tf.cast(images, tf.float32) / 255.0
        images = (images - IMAGENET_MEAN) / IMAGENET_STD
        return images

    # Step 4: Run inference in batches
    print(f"  Running inference on GPU...")
    all_coords = []
    all_scores = []

    num_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"Inference {split_name}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)

        # Get batch and normalize on GPU
        batch_images = all_images[start_idx:end_idx]
        batch_images_tf = normalize_batch(batch_images)

        # Run inference
        preds = model(batch_images_tf, training=False)

        # Handle tuple output
        if isinstance(preds, (list, tuple)):
            coords_batch = preds[0].numpy()
            scores_batch = preds[1].numpy()
        else:
            coords_batch = preds.numpy()
            scores_batch = np.ones((len(coords_batch), 1))

        all_coords.append(coords_batch)
        all_scores.append(scores_batch)

    # Concatenate all predictions
    all_coords = np.concatenate(all_coords, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    # Step 5: Compute metrics (vectorized)
    print(f"  Computing metrics...")

    # Fast bbox IoU for sorting
    ious_fast = compute_iou_batch(all_coords, all_gt)

    # Build results
    results = []
    for i in tqdm(range(n_samples), desc="Building results"):
        # Use exact polygon IoU for final results
        iou = compute_iou_polygon(all_coords[i], all_gt[i])
        err_mean, err_max = compute_pixel_error(all_coords[i], all_gt[i], img_size)

        results.append({
            'name': all_names[i],
            'split': split_name,
            'iou': iou,
            'iou_fast': ious_fast[i],
            'err_mean': err_mean,
            'err_max': err_max,
            'score': float(all_scores[i].flatten()[0]),
            'pred': all_coords[i].copy(),
            'gt': all_gt[i].copy(),
        })

    return results


def create_collage(results, data_root, output_path, title, n_images=20):
    """Create collage of worst cases."""
    tile_size = 400
    cols = 5
    rows = (n_images + cols - 1) // cols

    header_height = 60
    collage = Image.new('RGB', (cols * tile_size, rows * tile_size + header_height), (30, 30, 30))
    draw_header = ImageDraw.Draw(collage)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_small = font

    draw_header.text((10, 15), title, fill=(255, 255, 0), font=font)
    draw_header.text((10, 45), "GREEN = Ground Truth | RED = Prediction", fill=(200, 200, 200), font=font_small)

    for idx, r in enumerate(results[:n_images]):
        row = idx // cols
        col = idx % cols

        img_path = data_root / 'images' / r['name']
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert('RGB')
        img = img.resize((tile_size, tile_size), Image.BILINEAR)
        draw = ImageDraw.Draw(img)

        draw_quad(draw, r['gt'], (0, 255, 0), 3, tile_size)
        draw_quad(draw, r['pred'], (255, 0, 0), 3, tile_size)

        draw.rectangle([0, 0, tile_size, 55], fill=(0, 0, 0))
        draw.text((5, 3), f"IoU: {r['iou']:.3f} | Err: {r['err_mean']:.1f}px", fill=(255, 255, 0), font=font_small)
        draw.text((5, 20), f"[{r['split'].upper()}]", fill=(100, 200, 255), font=font_small)

        display_name = r['name']
        if len(display_name) > 45:
            display_name = display_name[:42] + "..."
        draw.text((5, 37), display_name, fill=(255, 255, 255), font=font_small)

        collage.paste(img, (col * tile_size, row * tile_size + header_height))

    collage.save(output_path, quality=95)
    print(f"Saved collage: {output_path}")


def _save_single_image(args):
    """Worker function for parallel image saving (must be top-level for pickle)."""
    r, gt, pred_coords, img_pil, out_path, img_size_output = args

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_small = font

    # Resize for output
    img_output = img_pil.resize((img_size_output, img_size_output), Image.BILINEAR)
    draw = ImageDraw.Draw(img_output)

    # Draw GT (green) and prediction (red)
    draw_quad(draw, gt, (0, 255, 0), 3, img_size_output)
    draw_quad(draw, pred_coords, (255, 0, 0), 3, img_size_output)

    # Info header
    header_height = 70
    info_img = Image.new('RGB', (img_size_output, img_size_output + header_height), (30, 30, 30))
    info_draw = ImageDraw.Draw(info_img)

    info_draw.text((10, 5), f"IoU: {r['iou']:.4f}  |  Error: {r['err_mean']:.2f}px (max: {r['err_max']:.2f}px)",
                  fill=(255, 255, 0), font=font)
    info_draw.text((10, 28), f"Split: {r['split'].upper()}  |  Score: {r['score']:.4f}",
                  fill=(100, 200, 255), font=font_small)

    display_name = r['name']
    if len(display_name) > 70:
        display_name = display_name[:67] + "..."
    info_draw.text((10, 48), display_name, fill=(200, 200, 200), font=font_small)

    info_img.paste(img_output, (0, header_height))

    # Legend
    info_draw.rectangle([img_size_output - 150, header_height + 5, img_size_output - 140, header_height + 15], fill=(0, 255, 0))
    info_draw.text((img_size_output - 135, header_height + 3), "GT", fill=(255, 255, 255), font=font_small)
    info_draw.rectangle([img_size_output - 80, header_height + 5, img_size_output - 70, header_height + 15], fill=(255, 0, 0))
    info_draw.text((img_size_output - 65, header_height + 3), "Pred", fill=(255, 255, 255), font=font_small)

    # Save as JPEG (faster than PNG)
    info_img.save(out_path, quality=85)
    return True


def save_individual_images_from_txt(txt_path, model, data_root, output_dir, img_size_model, img_size_output=512, batch_size=256):
    """
    Generate individual images with GT and prediction overlays from worst_cases.txt.
    Uses batched inference for speed.
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse txt file
    entries = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('=') or line.startswith('Split') or line.startswith('SUMMARY'):
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    entries.append({
                        'split': parts[0],
                        'name': parts[1],
                        'iou': float(parts[2]),
                        'err_mean': float(parts[3]),
                        'err_max': float(parts[4]),
                        'score': float(parts[5]) if len(parts) > 5 else 1.0,
                    })
                except:
                    continue

    print(f"Loaded {len(entries)} entries from {txt_path}")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_small = font

    # ImageNet normalization
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Step 1: Load all images and GT in parallel
    print(f"Loading {len(entries)} images...")

    def load_entry(r):
        img_path = data_root / 'images' / r['name']
        label_path = data_root / 'labels' / (Path(r['name']).stem + '.txt')
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
            img_pil = Image.open(img_path).convert('RGB')
            img_model = img_pil.resize((img_size_model, img_size_model), Image.BILINEAR)
            img_np = np.array(img_model, dtype=np.float32) / 255.0
            img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
            return (r, img_np, gt, img_pil)
        except:
            return None

    loaded = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        for result in tqdm(executor.map(load_entry, entries), total=len(entries), desc="Loading"):
            if result is not None:
                loaded.append(result)

    print(f"  Loaded {len(loaded)} valid images")

    # Step 2: Batch inference
    print(f"Running batch inference (batch_size={batch_size})...")
    all_preds = []
    n = len(loaded)

    for i in tqdm(range(0, n, batch_size), desc="Inference"):
        batch_data = loaded[i:i+batch_size]
        batch_imgs = np.stack([x[1] for x in batch_data], axis=0)

        preds = model(tf.constant(batch_imgs, dtype=tf.float32), training=False)
        if isinstance(preds, (list, tuple)):
            coords = preds[0].numpy()
        else:
            coords = preds.numpy()

        all_preds.extend(coords)

    # Step 3: Generate and save images in parallel using multiprocessing
    print(f"Generating {len(loaded)} output images (parallel)...")

    # Prepare data for parallel processing
    save_args = []
    for idx, (r, _, gt, img_pil) in enumerate(loaded):
        pred_coords = all_preds[idx]
        safe_name = Path(r['name']).stem.replace('/', '_').replace('\\', '_')
        out_path = str(output_dir / f"{idx+1:04d}_err{r['err_mean']:.1f}px_iou{r['iou']:.3f}_{safe_name}.jpg")
        save_args.append((r, gt, pred_coords, img_pil, out_path, img_size_output))

    # Use multiprocessing for CPU-bound image generation
    num_workers = min(mp.cpu_count(), 32)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(_save_single_image, save_args), total=len(save_args), desc="Saving"))

    print(f"Saved {len(loaded)} images to {output_dir}")

    # Create tar archive and delete images
    import subprocess
    import shutil
    tar_path = output_dir.parent / f"{output_dir.name}.tar.gz"
    print(f"Creating archive: {tar_path}")
    subprocess.run(['tar', '-czf', str(tar_path), '-C', str(output_dir.parent), output_dir.name], check=True)
    print(f"Removing directory: {output_dir}")
    shutil.rmtree(output_dir)
    print(f"Done! Archive: {tar_path}")


def save_results_txt(results, output_path):
    """Save detailed results to text file."""
    with open(output_path, 'w') as f:
        f.write(f"{'Split':<8} {'Filename':<60} {'IoU':>8} {'Err_mean':>10} {'Err_max':>10} {'Score':>8}\n")
        f.write("=" * 110 + "\n")

        for r in results:
            f.write(f"{r['split']:<8} {r['name']:<60} {r['iou']:>8.4f} {r['err_mean']:>10.2f} {r['err_max']:>10.2f} {r['score']:>8.4f}\n")

        f.write("\n" + "=" * 110 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 110 + "\n")

        for split in ['train', 'val']:
            split_results = [r for r in results if r['split'] == split]
            if split_results:
                ious = [r['iou'] for r in split_results]
                errs = [r['err_mean'] for r in split_results]
                f.write(f"\n{split.upper()} ({len(split_results)} images):\n")
                f.write(f"  IoU:  mean={np.mean(ious):.4f}, min={np.min(ious):.4f}, max={np.max(ious):.4f}\n")
                f.write(f"  Err:  mean={np.mean(errs):.2f}px, min={np.min(errs):.2f}px, max={np.max(errs):.2f}px\n")
                f.write(f"  IoU < 0.9: {sum(1 for i in ious if i < 0.9)} images\n")
                f.write(f"  IoU < 0.8: {sum(1 for i in ious if i < 0.8)} images\n")

    print(f"Saved results: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/workspace/checkpoints/mobilenetv2_256',
                        help='Checkpoint directory')
    parser.add_argument('--data_root', type=str, default='/workspace/doc-scanner-dataset',
                        help='Dataset root directory')
    parser.add_argument('--output_dir', type=str, default='/workspace',
                        help='Output directory for results')
    parser.add_argument('--n_worst', type=int, default=0,
                        help='Number of worst cases to show (0 = all above threshold)')
    parser.add_argument('--min_err', type=float, default=5.0,
                        help='Minimum pixel error threshold (show only cases with err >= this)')
    parser.add_argument('--sort_by', type=str, default='err', choices=['err', 'iou'],
                        help='Sort by: err (pixel error, descending) or iou (ascending)')
    parser.add_argument('--splits', type=str, default='train,val',
                        help='Comma-separated list of splits to evaluate')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for inference (1024 safe for A100 80GB)')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='Number of threads for image loading')
    parser.add_argument('--save_images', action='store_true',
                        help='Save individual worst case images with GT/pred overlay')
    parser.add_argument('--max_images', type=int, default=0,
                        help='Max number of individual images to save (0 = all)')
    parser.add_argument('--from_txt', type=str, default=None,
                        help='Generate images from existing worst_cases.txt (skip evaluation)')
    args = parser.parse_args()

    # Print GPU info
    print("\n" + "=" * 80)
    print("GPU-Optimized Evaluation")
    print("=" * 80)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU: {gpus[0].name}")
        try:
            gpu_info = tf.config.experimental.get_memory_info('GPU:0')
            print(f"  Memory available: {gpu_info.get('current', 0) / 1e9:.1f} GB")
        except:
            pass
    else:
        print("✗ No GPU found - running on CPU")

    print(f"XLA JIT: enabled")
    print("=" * 80)

    checkpoint_path = Path(args.checkpoint)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    # Load config
    config_path = checkpoint_path / 'config.json'
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    img_size = config.get('img_size', 224)
    print(f"\nModel config: img_size={img_size}, backbone={config.get('backbone')}")

    # Load model
    print("Loading model...")
    model = load_inference_model(
        checkpoint_path / 'best_model.weights.h5',
        backbone=config.get('backbone', 'mobilenetv2'),
        alpha=config.get('alpha', 0.35),
        fpn_ch=config.get('fpn_ch', 32),
        simcc_ch=config.get('simcc_ch', 96),
        img_size=img_size,
        num_bins=config.get('num_bins', img_size),
    )
    print("Model loaded")

    # Warmup inference (compile XLA kernels) - use small batch to avoid OOM
    print("Warming up GPU...")
    warmup_batch = min(64, args.batch_size)
    dummy_input = tf.random.normal([warmup_batch, img_size, img_size, 3])
    _ = model(dummy_input, training=False)
    del dummy_input
    print("Warmup done")

    # If --from_txt is provided, just generate images from existing txt
    if args.from_txt:
        txt_path = Path(args.from_txt)
        if not txt_path.exists():
            print(f"File not found: {txt_path}")
            sys.exit(1)

        images_dir = output_dir / 'worst_images'
        save_individual_images_from_txt(txt_path, model, data_root, images_dir, img_size)

        print(f"\n{'='*80}")
        print("DONE!")
        print(f"  Individual images: {images_dir}/")
        return

    # Evaluate splits
    all_results = []
    splits = [s.strip() for s in args.splits.split(',')]

    for split in splits:
        results = evaluate_split_gpu(model, data_root, split, img_size,
                                      args.batch_size, args.num_workers)
        all_results.extend(results)

    if not all_results:
        print("No results found!")
        sys.exit(1)

    # Filter by minimum error threshold
    if args.min_err > 0:
        filtered_results = [r for r in all_results if r['err_mean'] >= args.min_err]
        print(f"\nFiltered: {len(filtered_results)}/{len(all_results)} with err >= {args.min_err}px")
    else:
        filtered_results = all_results

    # Sort results
    if args.sort_by == 'err':
        filtered_results.sort(key=lambda x: -x['err_mean'])  # Descending by error
        sort_label = "by Pixel Error (descending)"
    else:
        filtered_results.sort(key=lambda x: x['iou'])  # Ascending by IoU
        sort_label = "by IoU (ascending)"

    # Limit number if specified
    if args.n_worst > 0:
        display_results = filtered_results[:args.n_worst]
    else:
        display_results = filtered_results

    # Print worst cases
    print(f"\n{'='*80}")
    print(f"WORST CASES {sort_label} (err >= {args.min_err}px)")
    print(f"Total: {len(display_results)} cases")
    print(f"{'='*80}")
    for i, r in enumerate(display_results):
        print(f"{i+1:3d}. [{r['split']:5s}] IoU={r['iou']:.4f} Err={r['err_mean']:6.2f}px  {r['name']}")

    # Create collage (max 100 images)
    collage_count = min(len(display_results), 100)
    collage_path = output_dir / 'worst_cases_collage.png'
    create_collage(display_results, data_root, collage_path,
                   f"Worst {collage_count} Cases (err >= {args.min_err}px)", collage_count)

    # Save results
    txt_path = output_dir / 'evaluation_results.txt'
    save_results_txt(all_results, txt_path)

    # Save filtered worst cases
    worst_txt_path = output_dir / 'worst_cases.txt'
    save_results_txt(display_results, worst_txt_path)

    # Save individual images if requested
    if args.save_images:
        images_dir = output_dir / 'worst_images'
        images_to_save = display_results
        if args.max_images > 0:
            images_to_save = display_results[:args.max_images]
        save_individual_images(images_to_save, data_root, images_dir)

    print(f"\n{'='*80}")
    print("DONE!")
    print(f"  Collage: {collage_path}")
    print(f"  Full results: {txt_path}")
    print(f"  Worst cases: {worst_txt_path}")
    if args.save_images:
        print(f"  Individual images: {output_dir / 'worst_images'}/")


if __name__ == "__main__":
    main()
