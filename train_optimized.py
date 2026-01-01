"""
Ultra-optimized training script for A100/high-end GPUs.

Key optimizations:
1. Mixed precision (FP16) - 2x throughput on tensor cores
2. XLA JIT compilation - fused kernels, reduced memory transfers
3. Full dataset in GPU memory as tf.Variable - zero copy overhead
4. Compiled train/val steps - no Python overhead
5. Large batch sizes (up to 2048 on A100 80GB)
6. Prefetch to GPU with tf.data
7. No tqdm overhead during training - pure speed

Usage:
    python train_optimized.py \
        --data_root /workspace/doc-scanner-dataset \
        --output_dir /workspace/checkpoints \
        --backbone mobilenetv2 \
        --img_size 256 \
        --batch_size 1024 \
        --epochs 300
"""

import argparse
import json
import os
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
# CRITICAL: Set environment BEFORE importing TensorFlow ops
# ============================================================================
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'  # Dedicated GPU threads
os.environ['TF_GPU_THREAD_COUNT'] = '2'  # Multiple GPU threads

# Enable mixed precision BEFORE importing model
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

from model import create_model, create_inference_model
from losses import gaussian_1d_targets
from dataset import tf_augment_batch


def setup_gpu():
    """Configure GPU for maximum performance."""
    print("\n" + "=" * 80)
    print("GPU Configuration")
    print("=" * 80)

    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("✗ No GPU found!")
        sys.exit(1)

    print(f"✓ Found {len(gpus)} GPU(s)")
    for gpu in gpus:
        print(f"  - {gpu.name}")

    # Enable memory growth to avoid OOM
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"  Warning: {e}")

    print(f"Mixed precision: {mixed_precision.global_policy().name}")
    print(f"XLA JIT: enabled")
    print("=" * 80 + "\n")


def load_single_image_fast(args):
    """Load a single image with label - optimized for threading."""
    name, data_root, img_size = args

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

    # Load and resize image
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
    """Load entire dataset into numpy arrays using parallel threading."""
    data_root = Path(data_root)

    # Find split file (same logic as generate_full_eval_csv.py)
    split_file = data_root / f"{split}.txt"
    if not split_file.exists():
        # Fallback to variants with negative
        for prefix in [f"{split}_with_negative_v2", f"{split}_with_negative"]:
            candidate = data_root / f"{prefix}.txt"
            if candidate.exists():
                split_file = candidate
                break

    if not split_file.exists():
        raise FileNotFoundError(f"No split file found for {split}")

    # Load image list
    with open(split_file) as f:
        image_names = [l.strip() for l in f if l.strip()]

    print(f"Loading {split}: {len(image_names)} images from {split_file.name}")
    print(f"  Using {num_workers} workers...")

    # Prepare args for parallel loading
    args_list = [(name, data_root, img_size) for name in image_names]

    # Load in parallel and collect results (same pattern as generate_full_eval_csv.py)
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(load_single_image_fast, args_list),
                          total=len(args_list), desc=f"Loading {split}", unit="img"):
            if result is not None:
                results.append(result)

    # Stack into numpy arrays (only valid samples)
    n_valid = len(results)
    print(f"  Stacking {n_valid} results into arrays...")

    # Pre-allocate and fill to avoid memory spikes from list comprehension
    images = np.empty((n_valid, img_size, img_size, 3), dtype=np.uint8)
    coords = np.empty((n_valid, 8), dtype=np.float32)
    has_doc = np.empty(n_valid, dtype=np.float32)

    for i, r in enumerate(results):
        images[i] = r[0]
        coords[i] = r[1]
        has_doc[i] = r[2]

    del results  # Free memory

    print(f"  Loaded {n_valid}/{len(image_names)} valid images ({images.nbytes / 1e9:.1f} GB)")
    return images, coords, has_doc


class GPUDataset:
    """
    Ultra-fast dataset that keeps data in GPU memory.

    Instead of tf.data which copies data every batch, we:
    1. Pre-normalize images on CPU (float32)
    2. Store as tf.constant on GPU
    3. Use tf.gather for batching - zero copy!
    """

    def __init__(self, images, coords, has_doc, batch_size, shuffle=True):
        self.n_samples = len(images)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches = self.n_samples // batch_size

        # Pre-normalize images (CPU)
        print("  Normalizing images...")
        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        images_norm = images.astype(np.float32) / 255.0
        images_norm = (images_norm - IMAGENET_MEAN) / IMAGENET_STD

        # Store as tf.constant (will be placed on GPU)
        print("  Transferring to GPU...")
        with tf.device('/GPU:0'):
            self.images = tf.constant(images_norm, dtype=tf.float32)
            self.coords = tf.constant(coords, dtype=tf.float32)
            self.has_doc = tf.constant(has_doc, dtype=tf.float32)

        # Pre-generate indices
        self.indices = np.arange(self.n_samples, dtype=np.int32)

        # Free CPU memory
        del images_norm
        gc.collect()

        print(f"  GPU dataset ready: {self.n_batches} batches of {batch_size}")

    def __len__(self):
        return self.n_batches

    def get_batches(self):
        """Generator yielding (indices_start, indices_end) for each batch."""
        if self.shuffle:
            np.random.shuffle(self.indices)

        for i in range(self.n_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            yield self.indices[start:end]

    @tf.function(jit_compile=True)
    def get_batch(self, indices):
        """Get a batch by indices - runs on GPU."""
        indices = tf.cast(indices, tf.int32)
        images = tf.gather(self.images, indices)
        coords = tf.gather(self.coords, indices)
        has_doc = tf.gather(self.has_doc, indices)
        return images, coords, has_doc


class Trainer:
    """
    Encapsulates training logic with compiled functions.

    All training operations are XLA-compiled for maximum speed.
    """

    def __init__(self, model, optimizer, img_size, sigma_px, tau, w_simcc, w_coord, w_score, augment=False):
        self.model = model
        self.optimizer = optimizer
        self.augment = augment

        # Store as tf.constant for XLA
        self.img_size = tf.constant(img_size, dtype=tf.int32)
        self.img_size_int = img_size  # Keep Python int for augmentation
        self.sigma_px = tf.constant(sigma_px, dtype=tf.float32)
        self.tau = tf.constant(tau, dtype=tf.float32)
        self.w_simcc = tf.constant(w_simcc, dtype=tf.float32)
        self.w_coord = tf.constant(w_coord, dtype=tf.float32)
        self.w_score = tf.constant(w_score, dtype=tf.float32)

    @tf.function
    def augment_batch(self, images, coords, has_doc):
        """Apply GPU augmentation to batch."""
        return tf_augment_batch(images, coords, has_doc, self.img_size_int, image_norm="imagenet")

    @tf.function(jit_compile=True)
    def train_step(self, images, coords_gt, has_doc):
        """Single training step - fully compiled."""
        with tf.GradientTape() as tape:
            outputs = self.model(images, training=True)

            # Extract outputs
            simcc_x = outputs["simcc_x"]
            simcc_y = outputs["simcc_y"]
            score_logit = outputs["score_logit"]
            coords_pred = outputs["coords"]

            # SimCC loss
            gt_coords_4x2 = tf.reshape(coords_gt, [-1, 4, 2])
            gt_x = gt_coords_4x2[:, :, 0]
            gt_y = gt_coords_4x2[:, :, 1]

            target_x = gaussian_1d_targets(gt_x, self.img_size, self.sigma_px)
            target_y = gaussian_1d_targets(gt_y, self.img_size, self.sigma_px)

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

            # Total loss (cast to float32 for stability)
            total_loss = tf.cast(
                self.w_simcc * loss_simcc + self.w_coord * loss_coord + self.w_score * loss_score,
                tf.float32
            )

            # Scale loss for mixed precision
            scaled_loss = self.optimizer.get_scaled_loss(total_loss)

        # Compute and apply gradients
        scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return total_loss

    @tf.function(jit_compile=True)
    def val_step(self, images, coords_gt, has_doc):
        """Validation step - returns predictions for IoU calculation."""
        outputs = self.model(images, training=False)
        return outputs["coords"]


def compute_metrics(coords_pred, coords_gt, has_doc, img_size):
    """Compute IoU and error metrics (numpy, runs on CPU)."""
    mask = has_doc > 0.5
    if mask.sum() == 0:
        return 0.0, 999.0

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
    mean_iou = float(ious.mean())

    # Corner error in pixels
    errors = np.abs(pred_pos - gt_pos) * img_size
    mean_error = float(errors.mean())

    return mean_iou, mean_error


def main():
    parser = argparse.ArgumentParser(description="Ultra-optimized training for A100 GPUs")

    # Data
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_split", type=str, default="train",
                        help="Training split file name (without .txt)")
    parser.add_argument("--val_split", type=str, default="val",
                        help="Validation split file name (without .txt)")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Custom name for experiment directory")

    # Model
    parser.add_argument("--backbone", type=str, default="mobilenetv2")
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--fpn_ch", type=int, default=32)
    parser.add_argument("--simcc_ch", type=int, default=96)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_bins", type=int, default=256)
    parser.add_argument("--tau", type=float, default=1.0)

    # Loss weights
    parser.add_argument("--sigma_px", type=float, default=3.0)
    parser.add_argument("--w_simcc", type=float, default=1.0)
    parser.add_argument("--w_coord", type=float, default=0.5)
    parser.add_argument("--w_score", type=float, default=0.5)

    # Training
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size (1024-2048 for A100 80GB)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr_patience", type=int, default=7)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--num_workers", type=int, default=64,
                        help="Number of threads for parallel image loading")
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation (GPU-accelerated)")

    args = parser.parse_args()

    # Setup GPU
    setup_gpu()

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.experiment_name:
        output_dir = Path(args.output_dir) / args.experiment_name
    else:
        output_dir = Path(args.output_dir) / f"{args.backbone}_{args.img_size}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Save config
    config = vars(args).copy()
    config["mixed_precision"] = True
    config["xla_jit"] = True
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ========================================================================
    # Load datasets
    # ========================================================================
    print("\n" + "=" * 80)
    print("Loading datasets...")
    print("=" * 80)

    train_images, train_coords, train_has_doc = load_dataset_fast(
        args.data_root, args.train_split, args.img_size, args.num_workers
    )
    val_images, val_coords, val_has_doc = load_dataset_fast(
        args.data_root, args.val_split, args.img_size, args.num_workers
    )

    # ========================================================================
    # Create GPU datasets
    # ========================================================================
    print("\n" + "=" * 80)
    print("Creating GPU datasets...")
    print("=" * 80)

    print("Train dataset:")
    train_ds = GPUDataset(train_images, train_coords, train_has_doc,
                          args.batch_size, shuffle=True)
    del train_images, train_coords, train_has_doc
    gc.collect()

    print("Val dataset:")
    val_ds = GPUDataset(val_images, val_coords, val_has_doc,
                        args.batch_size, shuffle=False)
    del val_images, val_coords, val_has_doc
    gc.collect()

    # ========================================================================
    # Create model
    # ========================================================================
    print("\n" + "=" * 80)
    print("Creating model...")
    print("=" * 80)

    model = create_model(
        backbone=args.backbone,
        alpha=args.alpha,
        fpn_ch=args.fpn_ch,
        simcc_ch=args.simcc_ch,
        img_size=args.img_size,
        num_bins=args.num_bins,
        tau=args.tau,
    )
    print(f"Parameters: {model.count_params():,}")

    # Optimizer with mixed precision loss scaling
    optimizer = keras.optimizers.AdamW(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    # Create trainer
    trainer = Trainer(
        model, optimizer, args.img_size, args.sigma_px, args.tau,
        args.w_simcc, args.w_coord, args.w_score, augment=args.augment
    )

    # ========================================================================
    # Warmup - compile XLA kernels
    # ========================================================================
    print("\nCompiling XLA kernels (warmup)...")
    warmup_indices = np.arange(min(64, args.batch_size), dtype=np.int32)
    images, coords, has_doc = train_ds.get_batch(warmup_indices)
    _ = trainer.train_step(images, coords, has_doc)
    _ = trainer.val_step(images, coords, has_doc)
    print("Warmup done - ready for full speed training!")

    # ========================================================================
    # Training loop
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"Starting training: {args.epochs} epochs, batch_size={args.batch_size}")
    if args.augment:
        print("Data augmentation: ENABLED (GPU-accelerated)")
    print("=" * 80)

    # Training state
    best_iou = 0.0
    best_epoch = 0
    current_lr = args.lr
    no_improve_count = 0
    lr_no_improve_count = 0
    history = {"train": [], "val": []}

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Warmup LR
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            optimizer.learning_rate.assign(warmup_lr)
            current_lr = warmup_lr

        # ====================================================================
        # Training
        # ====================================================================
        train_losses = []

        for batch_indices in train_ds.get_batches():
            images, coords, has_doc = train_ds.get_batch(batch_indices)
            # Apply augmentation on GPU if enabled
            if args.augment:
                images, coords = trainer.augment_batch(images, coords, has_doc)
            loss = trainer.train_step(images, coords, has_doc)
            train_losses.append(float(loss))

        avg_train_loss = np.mean(train_losses)

        # ====================================================================
        # Validation
        # ====================================================================
        all_preds = []
        all_coords = []
        all_has_doc = []

        for batch_indices in val_ds.get_batches():
            images, coords, has_doc = val_ds.get_batch(batch_indices)
            preds = trainer.val_step(images, coords, has_doc)
            all_preds.append(preds.numpy())
            all_coords.append(coords.numpy())
            all_has_doc.append(has_doc.numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_coords = np.concatenate(all_coords, axis=0)
        all_has_doc = np.concatenate(all_has_doc, axis=0)

        mean_iou, mean_error = compute_metrics(all_preds, all_coords, all_has_doc, args.img_size)

        epoch_time = time.time() - epoch_start
        samples_per_sec = (len(train_ds) * args.batch_size) / epoch_time

        # ====================================================================
        # Logging
        # ====================================================================
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"loss={avg_train_loss:.4f} | "
              f"IoU={mean_iou:.4f} | "
              f"err={mean_error:.1f}px | "
              f"LR={current_lr:.1e} | "
              f"{epoch_time:.1f}s ({samples_per_sec:.0f} img/s)")

        # Save history
        history["train"].append({"loss": avg_train_loss})
        history["val"].append({"mean_iou": mean_iou, "corner_error_px": mean_error})

        # ====================================================================
        # Checkpointing
        # ====================================================================
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_epoch = epoch + 1
            no_improve_count = 0
            lr_no_improve_count = 0

            # Save best model
            model.save_weights(str(output_dir / "best_model.weights.h5"))

            # Save inference model
            inference_model = create_inference_model(model)
            inference_model.save(output_dir / "best_model_inference.keras")

            print(f"  ★ New best IoU: {best_iou:.4f}")
        else:
            no_improve_count += 1
            lr_no_improve_count += 1

            # LR reduction
            if epoch >= args.warmup_epochs:
                if lr_no_improve_count >= args.lr_patience and current_lr > args.min_lr:
                    current_lr = max(current_lr * args.lr_factor, args.min_lr)
                    optimizer.learning_rate.assign(current_lr)
                    lr_no_improve_count = 0
                    print(f"  ↓ Reduced LR to {current_lr:.2e}")

        # Save history
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if no_improve_count >= args.patience:
            print(f"\n⚠ Early stopping at epoch {epoch + 1}")
            break

    # Save final model
    model.save_weights(str(output_dir / "final_model.weights.h5"))

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best epoch: {best_epoch} with IoU: {best_iou:.4f}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
