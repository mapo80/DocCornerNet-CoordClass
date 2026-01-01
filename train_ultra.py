"""
Ultra-optimized training script - cross-platform (A100/MPS/CPU).

Key optimizations:
1. Threading-based image loading (works everywhere, no semaphore issues)
2. Pre-allocated numpy arrays (no memory spikes)
3. tf.data pipeline with cache + prefetch
4. Mixed precision on CUDA
5. XLA JIT compilation on CUDA
6. Efficient training loop with minimal overhead

Usage:
    python train_ultra.py \
        --data_root /path/to/dataset \
        --output_dir /path/to/checkpoints \
        --backbone mobilenetv2 \
        --img_size 256 \
        --batch_size 512 \
        --epochs 100
"""

import argparse
import json
import os
import shutil
import sys
import gc
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from PIL import Image


# ============================================================================
# Platform detection and configuration
# ============================================================================
def _normalize_backbone_weights(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None
    return value


def _find_init_weights_path(value: str) -> Path:
    """
    Resolve a warm-start weights path.
      - If `value` is a directory, look for common weights filenames inside it.
      - If `value` is a file, return it as-is.
    """
    p = Path(value).expanduser()
    if p.is_dir():
        for candidate in [
            "best_model.weights.h5",
            "final_model.weights.h5",
            "latest_weights.h5",
            "best_iou_weights.h5",
            "final_weights.h5",
            # Student/distillation artifacts (if present)
            "best_student.weights.h5",
            "final_student.weights.h5",
        ]:
            cand = p / candidate
            if cand.exists():
                return cand
        raise FileNotFoundError(f"Cannot find a weights file in init_weights directory: {p}")
    if p.exists():
        return p
    raise FileNotFoundError(f"init_weights not found: {p}")


def setup_platform():
    """Configure platform for maximum performance."""
    gpus = tf.config.list_physical_devices('GPU')

    print("\n" + "=" * 80, flush=True)
    print("Platform Configuration", flush=True)
    print("=" * 80, flush=True)

    if gpus:
        # Check for NVIDIA
        try:
            from tensorflow.python.client import device_lib
            devices = device_lib.list_local_devices()
            is_nvidia = False
            for d in devices:
                if 'GPU' in d.device_type:
                    desc = d.physical_device_desc.lower()
                    if 'nvidia' in desc or 'cuda' in desc:
                        is_nvidia = True
                        print(f"  GPU: {d.physical_device_desc}", flush=True)
                        break

            if is_nvidia:
                # NVIDIA GPU optimizations
                os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
                os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Enable mixed precision
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy('mixed_float16')
                print(f"  Mixed precision: mixed_float16", flush=True)
                print(f"  XLA JIT: enabled", flush=True)
                print("=" * 80 + "\n", flush=True)
                return 'cuda'
        except Exception as e:
            print(f"  GPU detection warning: {e}")

    # Check for MPS (Apple Silicon)
    if sys.platform == 'darwin':
        print("  Using Metal Performance Shaders (MPS)", flush=True)
        print("  Mixed precision: float32", flush=True)
        print("=" * 80 + "\n", flush=True)
        return 'mps'

    # CPU fallback
    cpu_count = os.cpu_count() or 4
    try:
        tf.config.threading.set_intra_op_parallelism_threads(cpu_count)
        tf.config.threading.set_inter_op_parallelism_threads(cpu_count)
    except RuntimeError as exc:
        print(f"  Threading config skipped: {exc}", flush=True)
    print(f"  Using CPU with {cpu_count} threads", flush=True)
    print("=" * 80 + "\n", flush=True)
    return 'cpu'


# ============================================================================
# Fast threaded image loading (works everywhere)
# ============================================================================

def load_single_image(args):
    """Load single image - thread-safe."""
    name, data_root, img_size = args
    data_root = Path(data_root)

    image_dir = data_root / "images"
    negative_dir = data_root / "images-negative"
    label_dir = data_root / "labels"

    coords = np.zeros(8, dtype=np.float32)
    has_doc = 0.0

    # Determine image path and load label
    if name.startswith("negative_"):
        img_path = negative_dir / name
        has_doc = 0.0
    else:
        img_path = image_dir / name
        label_path = label_dir / f"{Path(name).stem}.txt"
        if label_path.exists():
            try:
                with open(label_path) as f:
                    line = f.readline().strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 9:
                            coords = np.array([float(x) for x in parts[1:9]], dtype=np.float32)
                            has_doc = 1.0
            except:
                pass

    if not img_path.exists():
        return None

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.resize((img_size, img_size), Image.BILINEAR)
            img_array = np.asarray(img, dtype=np.uint8).copy()
        return (img_array, coords, has_doc)
    except:
        return None


def load_dataset_fast(data_root, split, img_size, num_workers=64):
    """Load dataset using threading - fast and portable."""
    data_root = Path(data_root)

    # Find split file
    split_file = data_root / f"{split}.txt"
    if not split_file.exists():
        for suffix in ["_with_negative_v2", "_with_negative"]:
            candidate = data_root / f"{split}{suffix}.txt"
            if candidate.exists():
                split_file = candidate
                break

    if not split_file.exists():
        raise FileNotFoundError(f"No split file found for {split} in {data_root}")

    with open(split_file) as f:
        image_names = [l.strip() for l in f if l.strip()]

    n_images = len(image_names)
    print(f"Loading {split}: {n_images} images from {split_file.name}", flush=True)
    print(f"  Using {num_workers} threads...", flush=True)

    start_time = time.time()

    # Prepare arguments
    args_list = [(name, str(data_root), img_size) for name in image_names]

    # Load with progress
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(load_single_image, args_list),
                          total=n_images, desc=f"Loading {split}", unit="img"):
            if result is not None:
                results.append(result)

    load_time = time.time() - start_time
    n_valid = len(results)

    # Pre-allocate and fill arrays (avoids memory spike from np.stack)
    print(f"  Stacking {n_valid} results into arrays...", flush=True)
    stack_start = time.time()

    images = np.empty((n_valid, img_size, img_size, 3), dtype=np.uint8)
    coords = np.empty((n_valid, 8), dtype=np.float32)
    has_doc = np.empty(n_valid, dtype=np.float32)

    for i, (img, c, h) in enumerate(
        tqdm(results, total=n_valid, desc=f"Stacking {split}", unit="img")
    ):
        images[i] = img
        coords[i] = c
        has_doc[i] = h

    del results
    gc.collect()

    stack_time = time.time() - stack_start
    total_time = time.time() - start_time

    mem_gb = images.nbytes / 1e9
    print(f"  Loaded {n_valid}/{n_images} valid images ({mem_gb:.2f} GB)", flush=True)
    print(f"  Time: {load_time:.1f}s load + {stack_time:.1f}s stack = {total_time:.1f}s total", flush=True)
    print(f"  Speed: {n_valid / load_time:.0f} img/s", flush=True)

    return images, coords, has_doc


# ============================================================================
# tf.data based dataset
# ============================================================================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGENET_MEAN_TF = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
IMAGENET_STD_TF = tf.constant(IMAGENET_STD, dtype=tf.float32)


class FastDataset:
    """Optimized dataset - keeps data in numpy, normalizes in tf.data."""

    def __init__(self, images, coords, has_doc, batch_size, shuffle=True, drop_remainder=False, name="dataset"):
        self.n_samples = len(images)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder
        self.name = name
        self.chunk_size = max(self.batch_size * 4, 1024)

        # Store raw uint8 images (3GB instead of 12GB float32)
        self.images = images
        self.coords = coords.astype(np.float32)
        self.has_doc = has_doc.astype(np.float32)

        if drop_remainder:
            self.n_batches = self.n_samples // batch_size
        else:
            self.n_batches = (self.n_samples + batch_size - 1) // batch_size

        print("  Creating tf.data pipeline...", flush=True)
        dataset = self._build_base_dataset()
        if self.shuffle:
            dataset = dataset.shuffle(self.n_samples, reshuffle_each_iteration=True)

        dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
        dataset = dataset.map(self._normalize_batch, num_parallel_calls=tf.data.AUTOTUNE)
        self.dataset = dataset.prefetch(tf.data.AUTOTUNE)

        print(
            f"  Dataset ready: {self.n_batches} batches of {batch_size} "
            f"(drop_remainder={self.drop_remainder})",
            flush=True,
        )

    def _build_base_dataset(self):
        if self.n_samples == 0:
            with tf.device("/CPU:0"):
                return tf.data.Dataset.from_tensor_slices(
                    (self.images, self.coords, self.has_doc)
                )

        dataset = None
        for start in tqdm(
            range(0, self.n_samples, self.chunk_size),
            desc=f"Building {self.name} dataset",
            unit="chunk",
        ):
            end = min(start + self.chunk_size, self.n_samples)
            shard = (self.images[start:end], self.coords[start:end], self.has_doc[start:end])
            with tf.device("/CPU:0"):
                shard_ds = tf.data.Dataset.from_tensor_slices(shard)
            dataset = shard_ds if dataset is None else dataset.concatenate(shard_ds)

        return dataset

    @staticmethod
    def _normalize_batch(images, coords, has_doc):
        """Normalize a batch with TF ops."""
        images = tf.cast(images, tf.float32) / 255.0
        images = (images - IMAGENET_MEAN_TF) / IMAGENET_STD_TF
        return images, coords, has_doc

    def reshuffle(self):
        """No-op: shuffle handled by tf.data."""
        return

    def __len__(self):
        return self.n_batches


# ============================================================================
# Training logic
# ============================================================================

from model import create_model, create_inference_model
from losses import gaussian_1d_targets
from metrics import ValidationMetrics
from dataset import tf_augment_batch


class Trainer:
    """Efficient trainer with compiled functions."""

    def __init__(self, model, optimizer, img_size, sigma_px, tau,
                 w_simcc, w_coord, w_score, platform='cuda', augment=False):
        self.model = model
        self.optimizer = optimizer
        self.platform = platform
        self.use_mixed_precision = platform == 'cuda'
        self.augment = augment
        self.img_size = img_size

        # Pre-compute constants as tensors
        self.img_size_tf = tf.constant(img_size, dtype=tf.int32)
        self.img_size_float = tf.constant(float(img_size), dtype=tf.float32)
        self.sigma_px = tf.constant(sigma_px, dtype=tf.float32)
        self.tau = tf.constant(tau, dtype=tf.float32)
        self.w_simcc = tf.constant(w_simcc, dtype=tf.float32)
        self.w_coord = tf.constant(w_coord, dtype=tf.float32)
        self.w_score = tf.constant(w_score, dtype=tf.float32)

    @tf.function
    def augment_batch(self, images, coords, has_doc):
        """Apply augmentation to batch."""
        return tf_augment_batch(images, coords, has_doc, self.img_size, image_norm="imagenet")

    def _compute_loss(self, images, coords_gt, has_doc, training):
        """Compute total loss and its components."""
        outputs = self.model(images, training=training)

        # Cast to float32 for stable loss computation (model outputs are float16 with mixed precision)
        simcc_x = tf.cast(outputs["simcc_x"], tf.float32)
        simcc_y = tf.cast(outputs["simcc_y"], tf.float32)
        score_logit = tf.cast(outputs["score_logit"], tf.float32)
        coords_pred = tf.cast(outputs["coords"], tf.float32)

        # SimCC loss
        gt_coords_4x2 = tf.reshape(coords_gt, [-1, 4, 2])
        gt_x = gt_coords_4x2[:, :, 0]
        gt_y = gt_coords_4x2[:, :, 1]

        target_x = gaussian_1d_targets(gt_x, self.img_size_tf, self.sigma_px)
        target_y = gaussian_1d_targets(gt_y, self.img_size_tf, self.sigma_px)

        log_pred_x = tf.nn.log_softmax(simcc_x / self.tau, axis=-1)
        log_pred_y = tf.nn.log_softmax(simcc_y / self.tau, axis=-1)

        ce_x = -tf.reduce_sum(target_x * log_pred_x, axis=-1)
        ce_y = -tf.reduce_sum(target_y * log_pred_y, axis=-1)
        ce = tf.reduce_mean(ce_x + ce_y, axis=-1)
        loss_simcc = tf.reduce_sum(ce * has_doc) / (tf.reduce_sum(has_doc) + 1e-9)

        # Coord loss
        loss_per_coord = tf.abs(coords_pred - coords_gt)
        loss_per_sample = tf.reduce_mean(loss_per_coord, axis=-1)
        loss_coord = tf.reduce_sum(loss_per_sample * has_doc) / (tf.reduce_sum(has_doc) + 1e-9)

        # Score loss
        loss_score = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=has_doc[:, None],
            logits=score_logit
        )
        loss_score = tf.reduce_mean(loss_score)

        total_loss = self.w_simcc * loss_simcc + self.w_coord * loss_coord + self.w_score * loss_score

        return total_loss, coords_pred, loss_simcc, loss_coord, loss_score, score_logit

    def _batch_metrics(self, coords_pred, coords_gt, has_doc):
        """Compute IoU and corner error for positives in a batch."""
        img_size = tf.cast(self.img_size, tf.float32)
        has_doc_1d = tf.cast(has_doc, tf.float32)
        if len(has_doc_1d.shape) == 2:
            has_doc_1d = tf.squeeze(has_doc_1d, axis=-1)

        mask_bool = tf.cast(has_doc_1d, tf.bool)
        pred_pos = tf.boolean_mask(coords_pred, mask_bool)
        gt_pos = tf.boolean_mask(coords_gt, mask_bool)
        n_pos = tf.shape(pred_pos)[0]

        def compute_metrics():
            diff = tf.abs(pred_pos - gt_pos) * img_size
            corner_err = tf.reduce_mean(diff)

            pred_xy = tf.reshape(pred_pos, [-1, 4, 2])
            gt_xy = tf.reshape(gt_pos, [-1, 4, 2])

            pred_min = tf.reduce_min(pred_xy, axis=1)
            pred_max = tf.reduce_max(pred_xy, axis=1)
            gt_min = tf.reduce_min(gt_xy, axis=1)
            gt_max = tf.reduce_max(gt_xy, axis=1)

            inter_min = tf.maximum(pred_min, gt_min)
            inter_max = tf.minimum(pred_max, gt_max)
            inter_wh = tf.maximum(inter_max - inter_min, 0.0)
            inter_area = inter_wh[:, 0] * inter_wh[:, 1]

            pred_wh = pred_max - pred_min
            gt_wh = gt_max - gt_min
            pred_area = pred_wh[:, 0] * pred_wh[:, 1]
            gt_area = gt_wh[:, 0] * gt_wh[:, 1]

            union_area = pred_area + gt_area - inter_area + 1e-9
            iou = inter_area / union_area
            mean_iou = tf.reduce_mean(iou)

            return mean_iou, corner_err

        def zero_metrics():
            return tf.constant(0.0), tf.constant(0.0)

        return tf.cond(n_pos > 0, compute_metrics, zero_metrics)

    @tf.function
    def train_step(self, images, coords_gt, has_doc):
        """Training step."""
        with tf.GradientTape() as tape:
            (
                total_loss,
                coords_pred,
                loss_simcc,
                loss_coord,
                loss_score,
                _,
            ) = self._compute_loss(images, coords_gt, has_doc, training=True)
            if self.use_mixed_precision:
                if hasattr(self.optimizer, "scale_loss"):
                    scaled_loss = self.optimizer.scale_loss(total_loss)
                else:
                    scaled_loss = total_loss
            else:
                scaled_loss = total_loss

        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        iou, corner_err = self._batch_metrics(coords_pred, coords_gt, has_doc)
        return total_loss, loss_simcc, loss_coord, loss_score, iou, corner_err

    @tf.function
    def val_step(self, images, coords_gt, has_doc):
        """Validation step - returns predictions."""
        (
            total_loss,
            coords_pred,
            loss_simcc,
            loss_coord,
            loss_score,
            score_logit,
        ) = self._compute_loss(images, coords_gt, has_doc, training=False)
        return coords_pred, score_logit, total_loss, loss_simcc, loss_coord, loss_score


def compute_metrics(coords_pred, coords_gt, has_doc, img_size, score_logit=None):
    """Compute detailed IoU, error, and score metrics."""
    metrics = {}
    mask = has_doc > 0.5
    pos_count = int(mask.sum())
    neg_count = int((~mask).sum())
    metrics["pos_count"] = pos_count
    metrics["neg_count"] = neg_count

    if pos_count == 0:
        metrics.update({
            "mean_iou": 0.0,
            "median_iou": 0.0,
            "p90_iou": 0.0,
            "p95_iou": 0.0,
            "p99_iou": 0.0,
            "mean_err": 999.0,
            "median_err": 999.0,
            "p90_err": 999.0,
            "p95_err": 999.0,
            "p99_err": 999.0,
        })
    else:
        pred_pos = coords_pred[mask].reshape(-1, 4, 2)
        gt_pos = coords_gt[mask].reshape(-1, 4, 2)

        # Bounding box IoU (fast approximation)
        pred_min = pred_pos.min(axis=1)
        pred_max = pred_pos.max(axis=1)
        gt_min = gt_pos.min(axis=1)
        gt_max = gt_pos.max(axis=1)

        inter_min = np.maximum(pred_min, gt_min)
        inter_max = np.minimum(pred_max, gt_max)
        inter_wh = np.maximum(inter_max - inter_min, 0)
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]

        pred_area = (pred_max - pred_min).prod(axis=1)
        gt_area = (gt_max - gt_min).prod(axis=1)
        union_area = pred_area + gt_area - inter_area + 1e-9

        ious = inter_area / union_area
        metrics["mean_iou"] = float(ious.mean())
        metrics["median_iou"] = float(np.median(ious))
        metrics["p90_iou"] = float(np.percentile(ious, 90))
        metrics["p95_iou"] = float(np.percentile(ious, 95))
        metrics["p99_iou"] = float(np.percentile(ious, 99))

        errors = np.abs(pred_pos - gt_pos) * img_size
        error_per_sample = errors.reshape(errors.shape[0], -1).mean(axis=1)
        metrics["mean_err"] = float(error_per_sample.mean())
        metrics["median_err"] = float(np.median(error_per_sample))
        metrics["p90_err"] = float(np.percentile(error_per_sample, 90))
        metrics["p95_err"] = float(np.percentile(error_per_sample, 95))
        metrics["p99_err"] = float(np.percentile(error_per_sample, 99))

    if score_logit is not None:
        scores = 1.0 / (1.0 + np.exp(-np.clip(score_logit, -60.0, 60.0)))
        scores = scores.reshape(-1)
        labels = has_doc.reshape(-1)
        metrics["score_acc"] = float(((scores >= 0.5) == (labels >= 0.5)).mean())
        if pos_count > 0:
            metrics["score_pos_mean"] = float(scores[mask].mean())
        else:
            metrics["score_pos_mean"] = 0.0
        if neg_count > 0:
            metrics["score_neg_mean"] = float(scores[~mask].mean())
        else:
            metrics["score_neg_mean"] = 0.0

    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ultra-optimized cross-platform training")

    # Data
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--experiment_name", type=str, default=None)

    # Model
    parser.add_argument("--backbone", type=str, default="mobilenetv2")
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument(
        "--backbone_weights",
        type=str,
        default="imagenet",
        help="Backbone init weights ('imagenet' or None). None avoids downloads.",
    )
    parser.add_argument(
        "--init_weights",
        type=str,
        default=None,
        help=(
            "Optional warm-start weights (.weights.h5) or a checkpoint directory containing best_model.weights.h5. "
            "Useful for fine-tuning at a different img_size/num_bins."
        ),
    )
    parser.add_argument(
        "--init_partial",
        action="store_true",
        help="If strict init load fails, retry with by_name=True, skip_mismatch=True (HDF5 only).",
    )
    parser.add_argument("--fpn_ch", type=int, default=32)
    parser.add_argument("--simcc_ch", type=int, default=96)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_bins", type=int, default=256)
    parser.add_argument("--tau", type=float, default=1.0)

    # Loss
    parser.add_argument("--sigma_px", type=float, default=3.0)
    parser.add_argument("--w_simcc", type=float, default=1.0)
    parser.add_argument("--w_coord", type=float, default=0.5)
    parser.add_argument("--w_score", type=float, default=0.5)

    # Training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr_patience", type=int, default=7)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    # Loading
    parser.add_argument("--num_workers", type=int, default=64,
                        help="Threads for image loading")

    # Augmentation
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation")

    args = parser.parse_args()

    # Setup platform
    platform = setup_platform()
    use_mixed_precision = platform == 'cuda'

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.experiment_name:
        output_dir = Path(args.output_dir) / args.experiment_name
    else:
        output_dir = Path(args.output_dir) / f"{args.backbone}_{args.img_size}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}", flush=True)

    # Save config
    config = vars(args).copy()
    config["platform"] = platform
    config["mixed_precision"] = use_mixed_precision
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ========================================================================
    # Load datasets
    # ========================================================================
    print("\n" + "=" * 80, flush=True)
    print("Loading datasets...", flush=True)
    print("=" * 80, flush=True)

    train_images, train_coords, train_has_doc = load_dataset_fast(
        args.data_root, args.train_split, args.img_size, args.num_workers
    )
    val_images, val_coords, val_has_doc = load_dataset_fast(
        args.data_root, args.val_split, args.img_size, args.num_workers
    )

    # ========================================================================
    # Create tf.data datasets
    # ========================================================================
    print("\n" + "=" * 80, flush=True)
    print("Creating tf.data pipelines...", flush=True)
    print("=" * 80, flush=True)

    print("Train dataset:", flush=True)
    train_ds = FastDataset(train_images, train_coords, train_has_doc,
                           args.batch_size, shuffle=True, drop_remainder=True, name="train")
    del train_images, train_coords, train_has_doc
    gc.collect()

    print("Val dataset:", flush=True)
    val_ds = FastDataset(val_images, val_coords, val_has_doc,
                         args.batch_size, shuffle=False, drop_remainder=False, name="val")
    del val_images, val_coords, val_has_doc
    gc.collect()

    # ========================================================================
    # Create model
    # ========================================================================
    print("\n" + "=" * 80, flush=True)
    print("Creating model...", flush=True)
    print("=" * 80, flush=True)

    model = create_model(
        backbone=args.backbone,
        alpha=args.alpha,
        backbone_weights=_normalize_backbone_weights(args.backbone_weights),
        fpn_ch=args.fpn_ch,
        simcc_ch=args.simcc_ch,
        img_size=args.img_size,
        num_bins=args.num_bins,
        tau=args.tau,
    )
    print(f"Parameters: {model.count_params():,}", flush=True)

    # Optional warm-start (e.g. fine-tune at different img_size/num_bins).
    if args.init_weights:
        init_path = _find_init_weights_path(args.init_weights)
        print(f"\nLoading init weights from: {init_path}", flush=True)
        try:
            model.load_weights(str(init_path))
            print("✓ Loaded init weights (strict)", flush=True)
        except Exception as e:
            if not args.init_partial:
                raise
            print(f"Warning: strict init load failed: {e}", flush=True)
            print("Retrying with by_name=True, skip_mismatch=True...", flush=True)
            # Keras by_name loading only supports legacy HDF5 files ending in .h5/.hdf5.
            # Our checkpoints typically use the newer '*.weights.h5' naming; the file is
            # still HDF5, but Keras refuses by_name based on the filename. Create a
            # legacy-named copy for partial loading.
            init_for_by_name = init_path
            if init_path.name.endswith(".weights.h5"):
                legacy_name = init_path.name[: -len(".weights.h5")] + ".h5"
                legacy_path = output_dir / legacy_name
                if not legacy_path.exists():
                    shutil.copy2(init_path, legacy_path)
                init_for_by_name = legacy_path
            model.load_weights(str(init_for_by_name), by_name=True, skip_mismatch=True)
            print("✓ Loaded init weights (partial)", flush=True)

    # Optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    if use_mixed_precision:
        from tensorflow.keras import mixed_precision
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    # Trainer
    trainer = Trainer(
        model, optimizer, args.img_size, args.sigma_px, args.tau,
        args.w_simcc, args.w_coord, args.w_score,
        platform=platform, augment=args.augment
    )

    # ========================================================================
    # Warmup (compile XLA kernels)
    # ========================================================================
    print("\nCompiling (warmup)...", flush=True)
    for images, coords, has_doc in tqdm(
        train_ds.dataset.take(1),
        total=1,
        desc="Warmup",
        unit="batch",
    ):
        _ = trainer.train_step(images, coords, has_doc)
        _ = trainer.val_step(images, coords, has_doc)
    print("Warmup done!", flush=True)

    # ========================================================================
    # Training loop
    # ========================================================================
    print("\n" + "=" * 80, flush=True)
    print(f"Starting training: {args.epochs} epochs, batch_size={args.batch_size}", flush=True)
    if args.augment:
        print("Augmentation: ENABLED", flush=True)
    print("=" * 80, flush=True)

    best_iou = 0.0
    best_epoch = 0
    current_lr = args.lr
    no_improve_count = 0
    lr_no_improve_count = 0
    history = {"train": [], "val": []}

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Reshuffle training data for new epoch
        train_ds.reshuffle()

        # Warmup LR
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            optimizer.learning_rate.assign(warmup_lr)
            current_lr = warmup_lr

        print(f"\nEpoch {epoch + 1}/{args.epochs}", flush=True)

        # Training
        train_losses = []
        train_simcc = []
        train_coord = []
        train_score = []
        train_iou = []
        train_err = []
        train_pbar = tqdm(
            train_ds.dataset,
            total=len(train_ds),
            desc="  Train",
            unit="batch",
            ncols=100,
            leave=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]",
            ascii=True,
        )
        for images, coords, has_doc in train_pbar:
            if args.augment:
                images, coords = trainer.augment_batch(images, coords, has_doc)
            loss, loss_simcc, loss_coord, loss_score, batch_iou, batch_err = trainer.train_step(
                images, coords, has_doc
            )
            loss_val = float(loss)
            train_losses.append(loss_val)
            train_simcc.append(float(loss_simcc))
            train_coord.append(float(loss_coord))
            train_score.append(float(loss_score))
            train_iou.append(float(batch_iou))
            train_err.append(float(batch_err))
            train_pbar.set_postfix({
                "loss": f"{np.mean(train_losses):.4f}",
                "err": f"{np.mean(train_err):.1f}px",
                "iou": f"{np.mean(train_iou):.3f}",
            })

        avg_train_loss = float(np.mean(train_losses))
        avg_train_simcc = float(np.mean(train_simcc))
        avg_train_coord = float(np.mean(train_coord))
        avg_train_score = float(np.mean(train_score))
        avg_train_iou = float(np.mean(train_iou)) if train_iou else 0.0
        avg_train_err = float(np.mean(train_err)) if train_err else 0.0

        # Validation
        val_losses = []
        val_simcc = []
        val_coord = []
        val_score = []
        metrics = ValidationMetrics(img_size=args.img_size)
        val_pbar = tqdm(
            val_ds.dataset,
            total=len(val_ds),
            desc="  Val  ",
            unit="batch",
            ncols=100,
            leave=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]",
            ascii=True,
        )
        for images, coords, has_doc in val_pbar:
            preds, score_logit, v_loss, v_simcc, v_coord, v_score = trainer.val_step(
                images, coords, has_doc
            )
            val_losses.append(float(v_loss))
            val_simcc.append(float(v_simcc))
            val_coord.append(float(v_coord))
            val_score.append(float(v_score))
            score_pred = tf.sigmoid(score_logit).numpy()
            metrics.update(preds.numpy(), coords.numpy(), score_pred, has_doc.numpy())
            val_pbar.set_postfix({
                "loss": f"{np.mean(val_losses):.4f}",
            })

        val_metrics = metrics.compute()

        epoch_time = time.time() - epoch_start
        samples_per_sec = (len(train_ds) * args.batch_size) / epoch_time

        # Logging
        avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        avg_val_simcc = float(np.mean(val_simcc)) if val_simcc else 0.0
        avg_val_coord = float(np.mean(val_coord)) if val_coord else 0.0
        avg_val_score = float(np.mean(val_score)) if val_score else 0.0

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | LR={current_lr:.1e} | "
            f"{epoch_time:.1f}s ({samples_per_sec:.0f} img/s)",
            flush=True,
        )
        print(
            f"    Train: loss={avg_train_loss:.4f} "
            f"(simcc={avg_train_simcc:.4f} coord={avg_train_coord:.4f} score={avg_train_score:.4f})",
            flush=True,
        )
        print(
            f"           err={avg_train_err:.1f}px  IoU={avg_train_iou:.3f}",
            flush=True,
        )
        print(
            f"    Val:   loss={avg_val_loss:.4f} "
            f"(simcc={avg_val_simcc:.4f} coord={avg_val_coord:.4f} score={avg_val_score:.4f})",
            flush=True,
        )
        print(
            f"           err_mean={val_metrics['corner_error_px']:.1f}px  "
            f"err_p95={val_metrics['corner_error_p95_px']:.1f}px  "
            f"err_min={val_metrics['corner_error_min_px']:.1f}px  "
            f"err_max={val_metrics['corner_error_max_px']:.1f}px",
            flush=True,
        )
        print(
            f"           IoU={val_metrics['mean_iou']:.4f}  "
            f"R@90={val_metrics['recall_90']*100:.1f}%  "
            f"R@95={val_metrics['recall_95']*100:.1f}%  "
            f"R@99={val_metrics['recall_99']*100:.1f}%",
            flush=True,
        )
        print(
            f"           cls_acc={val_metrics['cls_accuracy']*100:.1f}%  "
            f"cls_f1={val_metrics['cls_f1']:.3f}",
            flush=True,
        )

        history["train"].append({
            "loss": avg_train_loss,
            "loss_simcc": avg_train_simcc,
            "loss_coord": avg_train_coord,
            "loss_score": avg_train_score,
            "iou": avg_train_iou,
            "corner_err_px": avg_train_err,
        })
        history["val"].append({
            "loss": avg_val_loss,
            "loss_simcc": avg_val_simcc,
            "loss_coord": avg_val_coord,
            "loss_score": avg_val_score,
            **val_metrics,
        })

        # Checkpointing
        if val_metrics["mean_iou"] > best_iou:
            best_iou = val_metrics["mean_iou"]
            best_epoch = epoch + 1
            no_improve_count = 0
            lr_no_improve_count = 0

            model.save_weights(str(output_dir / "best_model.weights.h5"))
            inference_model = create_inference_model(model)
            inference_model.save(output_dir / "best_model_inference.keras")

            print(f"  * New best IoU: {best_iou:.4f}", flush=True)
        else:
            no_improve_count += 1
            lr_no_improve_count += 1

            if epoch >= args.warmup_epochs:
                if lr_no_improve_count >= args.lr_patience and current_lr > args.min_lr:
                    current_lr = max(current_lr * args.lr_factor, args.min_lr)
                    optimizer.learning_rate.assign(current_lr)
                    lr_no_improve_count = 0
                    print(f"  -> Reduced LR to {current_lr:.2e}", flush=True)

        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if no_improve_count >= args.patience:
            print(f"\nEarly stopping at epoch {epoch + 1}", flush=True)
            break

    model.save_weights(str(output_dir / "final_model.weights.h5"))

    print("\n" + "=" * 80, flush=True)
    print("Training Complete!", flush=True)
    print("=" * 80, flush=True)
    print(f"Best epoch: {best_epoch} with IoU: {best_iou:.4f}", flush=True)
    print(f"Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
