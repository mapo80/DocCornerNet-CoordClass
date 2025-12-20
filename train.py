"""
Training script for DocCornerNetV3 (TensorFlow/Keras).

Features:
- Custom training loop with tqdm progress bars
- SimCC loss + Coordinate loss + Score loss
- Learning rate warmup + reduce on plateau
- Best model checkpointing (by IoU or corner error)
- Early stopping
"""

import argparse
import json
import math
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from model import create_model, create_inference_model
from losses import SimCCLoss, CoordLoss, gaussian_1d_targets
from dataset import create_dataset, load_split_file, preload_images_to_cache, tf_augment_batch
from metrics import ValidationMetrics


def print_device_info():
    """Print available devices and check GPU backend."""
    import platform

    print("\n" + "=" * 80)
    print("Device Information")
    print("=" * 80)

    # TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Platform: {platform.system()} {platform.machine()}")

    # List all physical devices
    physical_devices = tf.config.list_physical_devices()
    print(f"\nPhysical devices:")
    for device in physical_devices:
        print(f"  - {device.device_type}: {device.name}")

    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✓ GPU available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu.name}")

        # Detect backend type
        is_macos = platform.system() == "Darwin"
        if is_macos:
            print("  (Apple Metal/MPS backend)")
        else:
            # Check for CUDA
            try:
                cuda_version = tf.sysconfig.get_build_info().get('cuda_version', None)
                cudnn_version = tf.sysconfig.get_build_info().get('cudnn_version', None)
                if cuda_version:
                    print(f"  (CUDA {cuda_version}, cuDNN {cudnn_version})")
                else:
                    print("  (CUDA backend)")
            except Exception:
                print("  (CUDA backend)")
    else:
        print("\n✗ No GPU available - using CPU")

    print("=" * 80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DocCornerNetV3 model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument("--data_root", type=str,
                        default="../../datasets/official/doc-scanner-dataset-labeled",
                        help="Path to dataset root")
    parser.add_argument("--negative_dir", type=str, default="images-negative",
                        help="Negative images directory name")
    parser.add_argument("--outlier_list", type=str, default=None,
                        help="Path to outlier list file")
    parser.add_argument("--outlier_weight", type=float, default=3.0,
                        help="Weight multiplier for outlier sampling")

    # Caching
    parser.add_argument("--cache_images", action="store_true",
                        help="Pre-load images into RAM (faster, uses more memory)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory for persistent disk cache (e.g., ./cache)")
    parser.add_argument("--force_cache", action="store_true",
                        help="Force regeneration of disk cache")
    parser.add_argument("--fast_mode", action="store_true",
                        help="Ultra-fast mode: load all data as tensors, augment on GPU")

    # Model
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenetv3_small",
        choices=["mobilenetv2", "mobilenetv3_small", "mobilenetv3_large"],
        help="Backbone architecture",
    )
    parser.add_argument("--alpha", type=float, default=0.75,
                        help="Backbone width multiplier (alpha)")
    parser.add_argument(
        "--backbone_minimalistic",
        action="store_true",
        help="Use MobileNetV3 minimalistic variant (faster, more quantization-friendly)",
    )
    parser.add_argument(
        "--backbone_include_preprocessing",
        action="store_true",
        help="Enable built-in backbone preprocessing (expects raw uint8/0-255 inputs)",
    )
    parser.add_argument(
        "--backbone_weights",
        type=str,
        default="imagenet",
        help="Backbone init weights ('imagenet' or None). None avoids downloads.",
    )
    parser.add_argument("--fpn_ch", type=int, default=48,
                        help="FPN channels")
    parser.add_argument("--simcc_ch", type=int, default=128,
                        help="SimCC head channels")
    parser.add_argument("--num_bins", type=int, default=224,
                        help="Number of bins for coordinate classification")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Softmax temperature")

    # Loss
    parser.add_argument("--sigma_px", type=float, default=3.0,
                        help="Gaussian sigma for SimCC targets (larger = more tolerance)")
    parser.add_argument("--w_simcc", type=float, default=1.0,
                        help="SimCC loss weight")
    parser.add_argument("--w_coord", type=float, default=0.5,
                        help="Coordinate loss weight (direct supervision)")
    parser.add_argument("--w_score", type=float, default=0.5,
                        help="Score loss weight")

    # Training
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Warmup epochs")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--lr_patience", type=int, default=5,
                        help="LR reduction patience")
    parser.add_argument("--lr_factor", type=float, default=0.5,
                        help="LR reduction factor")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Minimum learning rate")
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation")

    # Output
    parser.add_argument("--output_dir", type=str, default="/workspace/checkpoints",
                        help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="v3_simcc",
                        help="Experiment name")

    return parser.parse_args()


def _normalize_backbone_weights(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None
    return value


class DocCornerNetV3Trainer:
    """
    Custom trainer with progress bar for each epoch.
    """

    def __init__(
        self,
        model: keras.Model,
        optimizer: keras.optimizers.Optimizer,
        img_size: int = 224,
        sigma_px: float = 2.0,
        tau: float = 1.0,
        w_simcc: float = 1.0,
        w_coord: float = 0.2,
        w_score: float = 1.0,
        fast_mode: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.img_size = img_size
        self.sigma_px = sigma_px
        self.tau = tau
        self.w_simcc = w_simcc
        self.w_coord = w_coord
        self.w_score = w_score
        self.fast_mode = fast_mode

        # Metrics for tracking
        self.train_loss = keras.metrics.Mean(name="loss")
        self.train_simcc_loss = keras.metrics.Mean(name="loss_simcc")
        self.train_coord_loss = keras.metrics.Mean(name="loss_coord")
        self.train_score_loss = keras.metrics.Mean(name="loss_score")
        self.train_iou = keras.metrics.Mean(name="iou")
        self.train_corner_err = keras.metrics.Mean(name="corner_err")

        self.val_loss = keras.metrics.Mean(name="val_loss")
        self.val_simcc_loss = keras.metrics.Mean(name="val_loss_simcc")
        self.val_coord_loss = keras.metrics.Mean(name="val_loss_coord")
        self.val_score_loss = keras.metrics.Mean(name="val_loss_score")

    def _compute_losses(self, outputs, targets):
        """Compute all losses."""
        simcc_x = outputs["simcc_x"]
        simcc_y = outputs["simcc_y"]
        score_logit = outputs["score_logit"]
        coords_pred = outputs["coords"]

        has_doc = tf.cast(targets["has_doc"], tf.float32)
        if len(has_doc.shape) == 2:
            has_doc = tf.squeeze(has_doc, axis=-1)
        coords_gt = tf.cast(targets["coords"], tf.float32)

        # SimCC loss (only positive samples)
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

        # Coord loss (only positive samples)
        loss_per_coord = tf.abs(coords_pred - coords_gt)
        loss_per_sample = tf.reduce_mean(loss_per_coord, axis=-1)
        loss_coord = tf.reduce_sum(loss_per_sample * has_doc) / (tf.reduce_sum(has_doc) + 1e-9)

        # Score loss (all samples)
        loss_score = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=has_doc[:, None],
            logits=score_logit
        )
        loss_score = tf.reduce_mean(loss_score)

        # Total loss
        total_loss = (
            self.w_simcc * loss_simcc +
            self.w_coord * loss_coord +
            self.w_score * loss_score
        )

        return {
            "total": total_loss,
            "simcc": loss_simcc,
            "coord": loss_coord,
            "score": loss_score,
        }

    def _compute_metrics(self, coords_pred, coords_gt, has_doc):
        """Compute IoU and corner error for positive samples."""
        img_size = 224.0

        mask_bool = tf.cast(has_doc, tf.bool)
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
    def train_step(self, images, targets, apply_augment=False):
        """Single training step."""
        coords = targets["coords"]
        has_doc = targets["has_doc"]
        # is_outlier may be present in fast_mode datasets
        is_outlier = targets["is_outlier"] if "is_outlier" in targets else None

        # Apply TF augmentation on GPU (fast_mode)
        if apply_augment:
            images, coords = tf_augment_batch(images, coords, has_doc, self.img_size, is_outlier)
            targets = {"coords": coords, "has_doc": has_doc}

        with tf.GradientTape() as tape:
            outputs = self.model(images, training=True)
            losses = self._compute_losses(outputs, targets)

        gradients = tape.gradient(losses["total"], self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return losses, outputs

    def train_epoch(self, train_ds, num_batches: int):
        """Train for one epoch with progress bar."""
        # Reset metrics
        self.train_loss.reset_state()
        self.train_simcc_loss.reset_state()
        self.train_coord_loss.reset_state()
        self.train_score_loss.reset_state()
        self.train_iou.reset_state()
        self.train_corner_err.reset_state()

        pbar = tqdm(
            train_ds,
            desc="  Train",
            total=num_batches,
            unit="batch",
            ncols=100,
            leave=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]",
            ascii="░█"
        )

        for images, targets in pbar:
            losses, outputs = self.train_step(images, targets, apply_augment=self.fast_mode)

            # Update metrics
            self.train_loss.update_state(losses["total"])
            self.train_simcc_loss.update_state(losses["simcc"])
            self.train_coord_loss.update_state(losses["coord"])
            self.train_score_loss.update_state(losses["score"])

            has_doc = tf.cast(targets["has_doc"], tf.float32)
            if len(has_doc.shape) == 2:
                has_doc = tf.squeeze(has_doc, axis=-1)
            iou, corner_err = self._compute_metrics(
                outputs["coords"], targets["coords"], has_doc
            )
            self.train_iou.update_state(iou)
            self.train_corner_err.update_state(corner_err)

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{float(self.train_loss.result()):.4f}",
                "err": f"{float(self.train_corner_err.result()):.1f}px",
                "iou": f"{float(self.train_iou.result()):.3f}",
            })

        return {
            "loss": float(self.train_loss.result()),
            "loss_simcc": float(self.train_simcc_loss.result()),
            "loss_coord": float(self.train_coord_loss.result()),
            "loss_score": float(self.train_score_loss.result()),
            "iou": float(self.train_iou.result()),
            "corner_err_px": float(self.train_corner_err.result()),
        }

    def validate(self, val_ds, num_batches: int):
        """Validate with progress bar and detailed metrics."""
        # Reset
        self.val_loss.reset_state()
        self.val_simcc_loss.reset_state()
        self.val_coord_loss.reset_state()
        self.val_score_loss.reset_state()

        # Full metrics accumulator
        metrics = ValidationMetrics(img_size=self.img_size)

        pbar = tqdm(
            val_ds,
            desc="  Val  ",
            total=num_batches,
            unit="batch",
            ncols=100,
            leave=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]",
            ascii="░█"
        )

        for images, targets in pbar:
            outputs = self.model(images, training=False)
            losses = self._compute_losses(outputs, targets)

            # Update loss metrics
            self.val_loss.update_state(losses["total"])
            self.val_simcc_loss.update_state(losses["simcc"])
            self.val_coord_loss.update_state(losses["coord"])
            self.val_score_loss.update_state(losses["score"])

            # Accumulate for detailed metrics
            coords_pred = outputs["coords"].numpy()
            score_pred = tf.sigmoid(outputs["score_logit"]).numpy()
            coords_gt = targets["coords"].numpy()
            has_doc = targets["has_doc"].numpy()

            metrics.update(coords_pred, coords_gt, score_pred, has_doc)

            pbar.set_postfix({
                "loss": f"{float(self.val_loss.result()):.4f}",
            })

        # Compute full metrics
        results = metrics.compute()

        return {
            "loss": float(self.val_loss.result()),
            "loss_simcc": float(self.val_simcc_loss.result()),
            "loss_coord": float(self.val_coord_loss.result()),
            "loss_score": float(self.val_score_loss.result()),
            **results,
        }


def main():
    args = parse_args()

    # Print device info first
    print_device_info()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DocCornerNetV3 Training")
    print("=" * 80)
    print(f"Output: {output_dir}")

    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Count samples
    data_root = Path(args.data_root)
    train_samples = load_split_file(str(data_root / "train_with_negative_v2.txt"))
    val_samples = load_split_file(str(data_root / "val_with_negative_v2.txt"))

    num_train_batches = math.ceil(len(train_samples) / args.batch_size)
    num_val_batches = math.ceil(len(val_samples) / args.batch_size)

    print(f"\nSamples: train={len(train_samples)}, val={len(val_samples)}")
    print(f"Batches: train={num_train_batches}, val={num_val_batches}")

    # Create datasets
    print("\nLoading datasets...")

    # Augmentation configs - stronger for better generalization
    augment_config = {
        "rotation_degrees": 10,       # 5 -> 10 degrees
        "scale_range": (0.8, 1.1),    # (0.9, 1.0) -> (0.8, 1.1) zoom in/out
        "brightness": 0.25,           # 0.2 -> 0.25
        "contrast": 0.25,             # 0.2 -> 0.25
        "saturation": 0.15,           # 0.1 -> 0.15
        "blur_prob": 0.2,             # 0.1 -> 0.2 (20% of samples)
        "blur_kernel": 3,
        "horizontal_flip": True,      # NEW: enable flip
    }

    # Stronger augmentation for outliers
    augment_config_outlier = {
        "rotation_degrees": 15,       # 8 -> 15 degrees
        "scale_range": (0.75, 1.15),  # wider range
        "brightness": 0.3,
        "contrast": 0.3,
        "saturation": 0.2,
        "blur_prob": 0.25,
        "blur_kernel": 5,
        "horizontal_flip": True,
    }

    # Pre-load images into cache if requested
    shared_cache = None
    if args.cache_images:
        data_root_path = Path(args.data_root)
        image_dir = data_root_path / "images"
        negative_dir = data_root_path / args.negative_dir

        # Load image lists for both splits
        all_images = []
        for split_name in ["train", "val"]:
            for prefix in [f"{split_name}_with_negative_v2", f"{split_name}_with_negative", split_name]:
                candidate = data_root_path / f"{prefix}.txt"
                if candidate.exists():
                    all_images.extend(load_split_file(str(candidate)))
                    break

        # Remove duplicates
        seen = set()
        unique_images = [img for img in all_images if img not in seen and not seen.add(img)]

        if unique_images:
            shared_cache = preload_images_to_cache(
                image_list=unique_images,
                image_dir=image_dir,
                negative_dir=negative_dir,
                img_size=args.img_size,
                cache_dir=args.cache_dir,
                force_cache=args.force_cache,
            )

    # Use fast_mode if cache is available and requested
    use_fast_mode = args.fast_mode and shared_cache is not None
    if args.fast_mode and shared_cache is None:
        print("Warning: --fast_mode requires --cache_images, falling back to standard mode")

    if use_fast_mode:
        print("\n⚡ Using FAST MODE: tensor loading + GPU augmentations")

    train_ds = create_dataset(
        data_root=args.data_root,
        split="train",
        img_size=args.img_size,
        batch_size=args.batch_size,
        shuffle=True,
        augment=args.augment if not use_fast_mode else False,  # Augment on GPU in fast_mode
        augment_config=augment_config if args.augment and not use_fast_mode else None,
        augment_config_outlier=augment_config_outlier if args.augment and not use_fast_mode else None,
        outlier_list=args.outlier_list,  # fast_mode uses this for is_outlier flag
        outlier_weight=args.outlier_weight,
        negative_dir=args.negative_dir,
        shared_cache=shared_cache,
        fast_mode=use_fast_mode,
    )
    val_ds = create_dataset(
        data_root=args.data_root,
        split="val",
        img_size=args.img_size,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
        negative_dir=args.negative_dir,
        shared_cache=shared_cache,
        fast_mode=use_fast_mode,
    )

    # Create model
    print("\nCreating model...")
    model = create_model(
        backbone=args.backbone,
        alpha=args.alpha,
        backbone_minimalistic=args.backbone_minimalistic,
        backbone_include_preprocessing=args.backbone_include_preprocessing,
        backbone_weights=_normalize_backbone_weights(args.backbone_weights),
        fpn_ch=args.fpn_ch,
        simcc_ch=args.simcc_ch,
        img_size=args.img_size,
        num_bins=args.num_bins,
        tau=args.tau,
    )
    print(f"Model parameters: {model.count_params():,}")

    if model.count_params() < 1_000_000:
        print("✓ Under 1M parameters target")
    else:
        print("✗ Over 1M parameters target")

    # Optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    # Trainer
    trainer = DocCornerNetV3Trainer(
        model=model,
        optimizer=optimizer,
        img_size=args.img_size,
        sigma_px=args.sigma_px,
        tau=args.tau,
        w_simcc=args.w_simcc,
        w_coord=args.w_coord,
        w_score=args.w_score,
        fast_mode=use_fast_mode,
    )

    # Training state
    best_iou = 0.0
    best_epoch = 0
    current_lr = args.lr
    no_improve_count = 0
    lr_no_improve_count = 0

    # History
    history = {"train": [], "val": []}

    print(f"\n{'=' * 80}")
    print(f"Starting training for {args.epochs} epochs...")
    print(f"{'=' * 80}")

    for epoch in range(args.epochs):
        # Warmup learning rate
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            optimizer.learning_rate.assign(warmup_lr)
            current_lr = warmup_lr

        print(f"\n{'─' * 80}")
        print(f"Epoch {epoch + 1}/{args.epochs}  (LR: {current_lr:.2e})")
        print(f"{'─' * 80}")

        # Train
        train_metrics = trainer.train_epoch(train_ds, num_train_batches)

        # Validate
        val_metrics = trainer.validate(val_ds, num_val_batches)

        # Print summary
        print(f"\n  Summary:")
        print(f"    Train: loss={train_metrics['loss']:.4f} "
              f"(simcc={train_metrics['loss_simcc']:.4f} "
              f"coord={train_metrics['loss_coord']:.4f} "
              f"score={train_metrics['loss_score']:.4f})")
        print(f"           err={train_metrics['corner_err_px']:.1f}px  "
              f"IoU={train_metrics['iou']:.3f}")
        print(f"    Val:   loss={val_metrics['loss']:.4f} "
              f"(simcc={val_metrics['loss_simcc']:.4f} "
              f"coord={val_metrics['loss_coord']:.4f} "
              f"score={val_metrics['loss_score']:.4f})")
        print(f"           err_mean={val_metrics['corner_error_px']:.1f}px  "
              f"err_p95={val_metrics['corner_error_p95_px']:.1f}px")
        print(f"           IoU={val_metrics['mean_iou']:.4f}  "
              f"R@90={val_metrics['recall_90']*100:.1f}%  "
              f"R@95={val_metrics['recall_95']*100:.1f}%")
        print(f"           cls_acc={val_metrics['cls_accuracy']*100:.1f}%  "
              f"cls_f1={val_metrics['cls_f1']:.3f}")

        # Record history
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Check for improvement (using IoU)
        if val_metrics["mean_iou"] > best_iou:
            best_iou = val_metrics["mean_iou"]
            best_epoch = epoch + 1
            no_improve_count = 0
            lr_no_improve_count = 0

            # Save best model (training version with full outputs)
            model.save(output_dir / "best_model.keras")
            model.save_weights(str(output_dir / "best_model.weights.h5"))

            # Save inference model (simplified output: coords, score)
            inference_model = create_inference_model(model)
            inference_model.save(output_dir / "best_model_inference.keras")
            print(f"  ★ Saved best model (IoU: {best_iou:.4f})")
        else:
            no_improve_count += 1
            lr_no_improve_count += 1

            # LR reduction (after warmup)
            if epoch >= args.warmup_epochs:
                if lr_no_improve_count >= args.lr_patience and current_lr > args.min_lr:
                    current_lr = max(current_lr * args.lr_factor, args.min_lr)
                    optimizer.learning_rate.assign(current_lr)
                    lr_no_improve_count = 0
                    print(f"  ↓ Reduced LR to {current_lr:.2e}")

        # Early stopping
        if no_improve_count >= args.patience:
            print(f"\n⚠ Early stopping at epoch {epoch + 1} "
                  f"(no improvement for {args.patience} epochs)")
            break

    # Save final model (training version)
    model.save(output_dir / "final_model.keras")
    model.save_weights(str(output_dir / "final_model.weights.h5"))

    # Save final inference model (simplified output: coords, score)
    inference_model = create_inference_model(model)
    inference_model.save(output_dir / "final_model_inference.keras")

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'=' * 80}")
    print("Training Complete!")
    print(f"{'=' * 80}")
    print(f"Best epoch: {best_epoch} with IoU: {best_iou:.4f}")
    print(f"Output saved to: {output_dir}")
    print(f"\nSaved models:")
    print(f"  - best_model.keras (training, dict output)")
    print(f"  - best_model_inference.keras (inference, tuple output)")
    print(f"  - final_model.keras (training, dict output)")
    print(f"  - final_model_inference.keras (inference, tuple output)")


if __name__ == "__main__":
    main()
