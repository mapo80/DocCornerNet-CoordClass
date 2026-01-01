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
import shutil
import sys
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
    parser.add_argument(
        "--input_norm",
        type=str,
        default="auto",
        choices=["auto", "imagenet", "zero_one", "raw255"],
        help=(
            "Input normalization for dataset images. "
            "Use 'auto' to pick raw255 when --backbone_include_preprocessing is set, otherwise imagenet."
        ),
    )
    parser.add_argument("--negative_dir", type=str, default="images-negative",
                        help="Negative images directory name")
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Train split name (resolved via *_with_negative_v2.txt fallback)",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="val",
        help="Val split name (resolved via *_with_negative_v2.txt fallback)",
    )
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
    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help=(
            "Resume from an existing run directory (restores model + optimizer state). "
            "When set, the run's config.json is used as the base configuration; any CLI "
            "flags you provide override it."
        ),
    )
    parser.add_argument(
        "--resume_epoch",
        type=int,
        default=None,
        help=(
            "Only used when --resume_dir is set but no TF checkpoint is found. "
            "Sets the starting epoch index (0-based) for logging and LR schedule. "
            "Example: --resume_epoch 42 will print 'Epoch 43/...'."
        ),
    )

    return parser.parse_args()


def _normalize_backbone_weights(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None
    return value


def _normalize_input_norm(value: str) -> str:
    if value is None:
        return "auto"
    value = value.strip().lower()
    if value in {"0_1", "01"}:
        return "zero_one"
    if value in {"0_255", "0255"}:
        return "raw255"
    return value


def _resolve_input_norm(value: str, backbone_include_preprocessing: bool) -> str:
    norm = _normalize_input_norm(value)
    if norm == "auto":
        return "raw255" if backbone_include_preprocessing else "imagenet"
    if norm not in {"imagenet", "zero_one", "raw255"}:
        raise ValueError(f"Unsupported input_norm='{value}'. Use: auto, imagenet, zero_one, raw255.")
    return norm


def _find_init_weights_path(value: str) -> Path:
    p = Path(value)
    if p.is_dir():
        for candidate in [
            p / "best_model.weights.h5",
            p / "final_model.weights.h5",
            p / "latest_weights.h5",
            p / "best_iou_weights.h5",
            p / "final_weights.h5",
        ]:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Cannot find a weights file in init_weights directory: {p}")
    if not p.exists():
        raise FileNotFoundError(f"init_weights not found: {p}")
    return p


def _cli_overrides_from_argv(argv: list[str]) -> set[str]:
    """
    Best-effort detection of which argparse options were explicitly provided.

    Used for `--resume_dir`: load the previous run config and only override values
    that the user actually specified on the CLI.
    """
    overrides: set[str] = set()
    for token in argv[1:]:
        if token.startswith("--"):
            overrides.add(token.split("=", 1)[0])
    return overrides


def _load_resume_config(resume_dir: Path) -> dict:
    config_path = resume_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"resume_dir missing config.json: {config_path}")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config.json in resume_dir: {config_path}")
    return cfg


def _apply_resume_config(args, resume_cfg: dict, cli_overrides: set[str]):
    """
    Merge resume config into args, keeping any explicitly provided CLI values.
    """
    for key, value in resume_cfg.items():
        flag = f"--{key}"
        if flag in cli_overrides:
            continue
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def _find_split_file(data_root: Path, split: str) -> Path:
    """Match the same split-file fallback logic as dataset.create_dataset()."""
    for prefix in (f"{split}_with_negative_v2", f"{split}_with_negative", split):
        candidate = data_root / f"{prefix}.txt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No split file found for split='{split}' in {data_root}")


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
        augment: bool = False,
        image_norm: str = "imagenet",
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
        self.augment = bool(augment)
        self.image_norm = str(image_norm)

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
        img_size = tf.cast(self.img_size, tf.float32)

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
            images, coords = tf_augment_batch(
                images,
                coords,
                has_doc,
                self.img_size,
                is_outlier,
                image_norm=self.image_norm,
            )
            targets = {"coords": coords, "has_doc": has_doc}

        with tf.GradientTape() as tape:
            outputs = self.model(images, training=True)
            losses = self._compute_losses(outputs, targets)

        gradients = tape.gradient(losses["total"], self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        has_doc_1d = tf.cast(targets["has_doc"], tf.float32)
        if len(has_doc_1d.shape) == 2:
            has_doc_1d = tf.squeeze(has_doc_1d, axis=-1)

        iou, corner_err = self._compute_metrics(outputs["coords"], targets["coords"], has_doc_1d)

        return losses, iou, corner_err

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
            losses, iou, corner_err = self.train_step(
                images,
                targets,
                apply_augment=(self.fast_mode and self.augment),
            )

            # Update metrics
            self.train_loss.update_state(losses["total"])
            self.train_simcc_loss.update_state(losses["simcc"])
            self.train_coord_loss.update_state(losses["coord"])
            self.train_score_loss.update_state(losses["score"])

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

    # Resolve input normalization early so it is persisted in config.json.
    args.input_norm = _resolve_input_norm(args.input_norm, args.backbone_include_preprocessing)

    cli_overrides = _cli_overrides_from_argv(sys.argv)
    resume_dir = Path(args.resume_dir).expanduser().resolve() if args.resume_dir else None

    # Resume mode: load previous config and continue in-place.
    if resume_dir is not None:
        resume_cfg = _load_resume_config(resume_dir)
        args = _apply_resume_config(args, resume_cfg, cli_overrides)
        output_dir = resume_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n↻ Resume enabled: {output_dir}")
    else:
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f"{args.experiment_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DocCornerNetV3 Training")
    print("=" * 80)
    print(f"Output: {output_dir}")

    # Save config (new runs only). For resume, keep the original config.json untouched.
    if resume_dir is None:
        config = vars(args)
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
    else:
        # Save resume CLI overrides for traceability.
        overrides_path = output_dir / f"resume_overrides_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        effective = vars(args)
        overrides = {k: effective[k] for k in effective.keys() if f"--{k}" in cli_overrides}
        overrides["resume_dir"] = str(output_dir)
        with open(overrides_path, "w") as f:
            json.dump(overrides, f, indent=2)

    # Count samples
    data_root = Path(args.data_root)
    train_split_file = _find_split_file(data_root, args.train_split)
    val_split_file = _find_split_file(data_root, args.val_split)
    train_samples = load_split_file(str(train_split_file))
    val_samples = load_split_file(str(val_split_file))

    num_train_batches = math.ceil(len(train_samples) / args.batch_size)
    num_val_batches = math.ceil(len(val_samples) / args.batch_size)

    print(f"\nSamples: train={len(train_samples)}, val={len(val_samples)}")
    print(f"Batches: train={num_train_batches}, val={num_val_batches}")

    # Create datasets
    print("\nLoading datasets...")
    print(f"Input normalization: {args.input_norm}")

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
        for split_name in [args.train_split, args.val_split]:
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
        split=args.train_split,
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
        image_norm=args.input_norm,
    )
    val_ds = create_dataset(
        data_root=args.data_root,
        split=args.val_split,
        img_size=args.img_size,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
        negative_dir=args.negative_dir,
        shared_cache=shared_cache,
        fast_mode=use_fast_mode,
        image_norm=args.input_norm,
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

    # Resume checkpoint (model + optimizer + training counters)
    ckpt_dir = output_dir / "tf_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = tf.train.latest_checkpoint(str(ckpt_dir))

    ckpt_epoch = tf.Variable(0, dtype=tf.int64, name="epoch")
    ckpt_best_iou = tf.Variable(0.0, dtype=tf.float32, name="best_iou")
    ckpt_best_epoch = tf.Variable(0, dtype=tf.int64, name="best_epoch")
    ckpt_no_improve = tf.Variable(0, dtype=tf.int64, name="no_improve_count")
    ckpt_lr_no_improve = tf.Variable(0, dtype=tf.int64, name="lr_no_improve_count")
    ckpt_current_lr = tf.Variable(float(args.lr), dtype=tf.float32, name="current_lr")

    # Ensure optimizer slot variables exist before restore (Keras 3).
    try:
        optimizer.build(model.trainable_variables)
    except Exception:
        pass

    ckpt = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=ckpt_epoch,
        best_iou=ckpt_best_iou,
        best_epoch=ckpt_best_epoch,
        no_improve=ckpt_no_improve,
        lr_no_improve=ckpt_lr_no_improve,
        current_lr=ckpt_current_lr,
    )
    manager = tf.train.CheckpointManager(ckpt, str(ckpt_dir), max_to_keep=3)

    did_restore = False
    if resume_dir is not None and latest_ckpt_path:
        print(f"\n↻ Restoring from checkpoint: {latest_ckpt_path}")
        ckpt.restore(latest_ckpt_path).expect_partial()
        did_restore = True
        try:
            optimizer.learning_rate.assign(float(ckpt_current_lr.numpy()))
        except Exception:
            pass
    elif resume_dir is not None:
        print("\n↻ Resume requested but no checkpoint found.")

    # Optional warm-start (only when not restored from a checkpoint)
    if (not did_restore) and (resume_dir is not None) and ("--init_weights" not in cli_overrides):
        # If resuming but the previous run predates TF checkpointing, default to the
        # run's own best/final weights (if present) so the user gets a sensible
        # "pseudo-resume" experience.
        if (output_dir / "best_model.weights.h5").exists() or (output_dir / "final_model.weights.h5").exists():
            args.init_weights = str(output_dir)

    if (not did_restore) and args.init_weights:
        init_path = _find_init_weights_path(args.init_weights)
        print(f"\nLoading init weights from: {init_path}")
        try:
            model.load_weights(str(init_path))
            print("✓ Loaded init weights (strict)")
        except Exception as e:
            if not args.init_partial:
                raise
            print(f"Warning: strict init load failed: {e}")
            print("Retrying with by_name=True, skip_mismatch=True...")
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
            print("✓ Loaded init weights (partial)")

        # If we are in resume mode without a TF checkpoint, optionally bump the
        # epoch counter for correct logging/LR schedule. This does NOT restore
        # the optimizer state (not available).
        if resume_dir is not None:
            if args.resume_epoch is not None:
                try:
                    ckpt_epoch.assign(int(args.resume_epoch))
                except Exception:
                    pass

            # Create an initial checkpoint so future resumes work, even if we crash mid-epoch.
            ckpt_best_iou.assign(float(best_iou) if "best_iou" in locals() else 0.0)
            ckpt_best_epoch.assign(int(best_epoch) if "best_epoch" in locals() else 0)
            ckpt_no_improve.assign(int(no_improve_count) if "no_improve_count" in locals() else 0)
            ckpt_lr_no_improve.assign(int(lr_no_improve_count) if "lr_no_improve_count" in locals() else 0)
            ckpt_current_lr.assign(float(current_lr) if "current_lr" in locals() else float(args.lr))
            manager.save()

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
        augment=args.augment,
        image_norm=args.input_norm,
    )

    # Training state
    best_iou = float(ckpt_best_iou.numpy()) if did_restore else 0.0
    best_epoch = int(ckpt_best_epoch.numpy()) if did_restore else 0
    current_lr = float(ckpt_current_lr.numpy()) if did_restore else float(args.lr)
    no_improve_count = int(ckpt_no_improve.numpy()) if did_restore else 0
    lr_no_improve_count = int(ckpt_lr_no_improve.numpy()) if did_restore else 0
    # In resume mode without a TF checkpoint, allow resume_epoch to set the counter
    # for correct logging/LR schedule (but optimizer state is fresh).
    if (not did_restore) and (resume_dir is not None) and (args.resume_epoch is not None):
        start_epoch = int(args.resume_epoch)
    else:
        start_epoch = int(ckpt_epoch.numpy()) if did_restore else 0

    # History
    history_path = output_dir / "history.json"
    history = {"train": [], "val": []}
    if did_restore and history_path.exists():
        try:
            with open(history_path, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and "train" in loaded and "val" in loaded:
                history = loaded
        except Exception:
            pass

    print(f"\n{'=' * 80}")
    if did_restore:
        print(
            f"Resuming at epoch {start_epoch + 1}/{args.epochs} "
            f"(best_iou={best_iou:.4f} @ epoch {best_epoch}, lr={current_lr:.2e})"
        )
    else:
        print(f"Starting training for {args.epochs} epochs...")
    print(f"{'=' * 80}")

    if start_epoch >= int(args.epochs):
        print(f"\nResume epoch ({start_epoch}) >= epochs ({args.epochs}). Nothing to do.")
        return

    for epoch in range(start_epoch, int(args.epochs)):
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
              f"err_p95={val_metrics['corner_error_p95_px']:.1f}px  "
              f"err_min={val_metrics['corner_error_min_px']:.1f}px  "
              f"err_max={val_metrics['corner_error_max_px']:.1f}px")
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

        # Persist resume state each epoch (so crashes can resume cleanly)
        ckpt_epoch.assign(epoch + 1)  # number of completed epochs
        ckpt_best_iou.assign(float(best_iou))
        ckpt_best_epoch.assign(int(best_epoch))
        ckpt_no_improve.assign(int(no_improve_count))
        ckpt_lr_no_improve.assign(int(lr_no_improve_count))
        ckpt_current_lr.assign(float(current_lr))
        manager.save()

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

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
    with open(history_path, "w") as f:
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
