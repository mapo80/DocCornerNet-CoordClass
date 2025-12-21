"""
Train a smaller DocCornerNetV3 "student" model using knowledge distillation.

The student is trained with a mix of:
- Hard targets (ground truth): SimCC (Gaussian soft labels) + coord L1 + score BCE
- Soft targets (teacher): SimCC logit distillation (KL/CE) + optional coord/score distillation

Typical usage:
  python train_student.py \\
    --teacher_weights ./checkpoints/best_model.weights.h5 \\
    --data_root /path/to/dataset \\
    --student_alpha 0.5 --student_fpn_ch 32 --student_simcc_ch 96 \\
    --epochs 60 --batch_size 64 --augment --cache_images --fast_mode
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from model import create_model, create_inference_model
from losses import gaussian_1d_targets
from dataset import (
    create_dataset,
    load_split_file,
    preload_images_to_cache,
    tf_augment_batch,
)
from metrics import ValidationMetrics


def print_device_info():
    """Print available devices (same as train.py)."""
    import platform

    print("\n" + "=" * 80)
    print("Device Information")
    print("=" * 80)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Platform: {platform.system()} {platform.machine()}")

    physical_devices = tf.config.list_physical_devices()
    print("\nPhysical devices:")
    for device in physical_devices:
        print(f"  - {device.device_type}: {device.name}")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\n✓ GPU available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu.name}")
        if platform.system() == "Darwin":
            print("  (Apple Metal/MPS backend)")
    else:
        print("\n✗ No GPU available - using CPU")

    print("=" * 80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a distilled student for DocCornerNetV3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--data_root",
        type=str,
        default="../../datasets/official/doc-scanner-dataset-labeled",
        help="Path to dataset root",
    )
    parser.add_argument(
        "--input_norm",
        type=str,
        default="auto",
        choices=["auto", "imagenet", "zero_one", "raw255"],
        help=(
            "Input normalization for dataset images. "
            "Use 'auto' to pick the teacher config value when available, otherwise "
            "raw255 when teacher includes preprocessing, else imagenet."
        ),
    )
    parser.add_argument(
        "--negative_dir",
        type=str,
        default="images-negative",
        help="Negative images directory name",
    )
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
    parser.add_argument("--outlier_list", type=str, default=None, help="Path to outlier list file")
    parser.add_argument("--outlier_weight", type=float, default=3.0, help="Outlier sampling weight")

    # Caching
    parser.add_argument("--cache_images", action="store_true", help="Pre-load images into RAM")
    parser.add_argument("--cache_dir", type=str, default=None, help="Persistent cache dir")
    parser.add_argument("--force_cache", action="store_true", help="Force cache regeneration")
    parser.add_argument("--fast_mode", action="store_true", help="GPU augmentations (requires cache)")

    # Teacher (weights + optional config auto-load)
    parser.add_argument(
        "--teacher_weights",
        type=str,
        default="./checkpoints/best_model.weights.h5",
        help="Path to teacher weights (.weights.h5)",
    )
    parser.add_argument(
        "--teacher_onnx",
        type=str,
        default=None,
        help=(
            "Optional ONNX teacher path. If set, distillation uses the ONNX teacher and "
            "--teacher_weights/config/backbone args are ignored."
        ),
    )
    parser.add_argument(
        "--teacher_onnx_input_norm",
        type=str,
        default="zero_one",
        choices=["imagenet", "zero_one", "raw255", "m1p1"],
        help=(
            "Preprocessing expected by the ONNX teacher input. "
            "For the provided FastViT heatmap teachers this is typically 'zero_one' (img/255)."
        ),
    )
    parser.add_argument(
        "--teacher_onnx_provider",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="onnxruntime provider selection (best-effort).",
    )
    parser.add_argument(
        "--teacher_config",
        type=str,
        default=None,
        help="Optional config.json for teacher (auto-detected next to weights)",
    )
    parser.add_argument("--teacher_alpha", type=float, default=0.75)
    parser.add_argument("--teacher_fpn_ch", type=int, default=48)
    parser.add_argument("--teacher_simcc_ch", type=int, default=128)
    parser.add_argument("--teacher_tau", type=float, default=1.0)
    parser.add_argument(
        "--teacher_backbone",
        type=str,
        default="mobilenetv3_small",
        choices=["mobilenetv2", "mobilenetv3_small", "mobilenetv3_large"],
        help="Teacher backbone architecture",
    )
    parser.add_argument(
        "--teacher_backbone_minimalistic",
        action="store_true",
        help="Use MobileNetV3 minimalistic variant for teacher",
    )
    parser.add_argument(
        "--teacher_backbone_include_preprocessing",
        action="store_true",
        help="Enable built-in backbone preprocessing for teacher (expects raw uint8/0-255 inputs)",
    )
    parser.add_argument(
        "--teacher_backbone_weights",
        type=str,
        default=None,
        help="Backbone init weights for teacher ('imagenet' or None). None avoids downloads.",
    )

    # Shared I/O shapes (must match teacher)
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--num_bins", type=int, default=224, help="Bins for SimCC (usually img_size)")
    parser.add_argument("--tau", type=float, default=1.0, help="Student soft-argmax temperature")

    # Student architecture
    parser.add_argument("--student_alpha", type=float, default=0.5, help="Student backbone width")
    parser.add_argument(
        "--student_backbone",
        type=str,
        default="mobilenetv3_small",
        choices=["mobilenetv2", "mobilenetv3_small", "mobilenetv3_large"],
        help="Student backbone architecture",
    )
    parser.add_argument(
        "--student_backbone_minimalistic",
        action="store_true",
        help="Use MobileNetV3 minimalistic variant for student",
    )
    parser.add_argument(
        "--student_backbone_include_preprocessing",
        action="store_true",
        help="Enable built-in backbone preprocessing for student (expects raw uint8/0-255 inputs)",
    )
    parser.add_argument("--student_fpn_ch", type=int, default=32, help="Student FPN channels")
    parser.add_argument("--student_simcc_ch", type=int, default=96, help="Student head channels")
    parser.add_argument(
        "--student_init_weights",
        type=str,
        default=None,
        help=(
            "Optional warm-start weights (.weights.h5) or a checkpoint directory containing best_model.weights.h5 "
            "(strict load)."
        ),
    )
    parser.add_argument(
        "--student_init_partial",
        action="store_true",
        help="If strict student init load fails, retry with by_name=True, skip_mismatch=True (HDF5 only).",
    )
    parser.add_argument(
        "--student_backbone_weights",
        type=str,
        default="imagenet",
        help="Backbone init weights for student ('imagenet' or None).",
    )

    # Hard losses (ground truth)
    parser.add_argument("--sigma_px", type=float, default=3.0, help="Gaussian sigma for targets")
    parser.add_argument("--w_simcc", type=float, default=1.0, help="Hard SimCC loss weight")
    parser.add_argument("--w_coord", type=float, default=0.5, help="Hard coord loss weight")
    parser.add_argument("--w_score", type=float, default=0.5, help="Hard score loss weight")

    # Distillation
    parser.add_argument("--distill_tau", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--w_distill_simcc", type=float, default=1.0, help="Distill SimCC weight")
    parser.add_argument("--w_distill_coord", type=float, default=0.0, help="Distill coord weight")
    parser.add_argument("--w_distill_score", type=float, default=0.1, help="Distill score weight")

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--augment", action="store_true")

    # Output
    parser.add_argument("--output_dir", type=str, default="./checkpoints_student")
    parser.add_argument("--experiment_name", type=str, default="student_distill")

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


def _resolve_input_norm(value: str, teacher_cfg: dict, teacher_include_preprocessing: bool) -> str:
    norm = _normalize_input_norm(value)
    if norm == "auto":
        cfg_norm = _normalize_input_norm(teacher_cfg.get("input_norm")) if teacher_cfg else "auto"
        if cfg_norm != "auto":
            return cfg_norm
        return "raw255" if teacher_include_preprocessing else "imagenet"
    if norm not in {"imagenet", "zero_one", "raw255"}:
        raise ValueError(f"Unsupported input_norm='{value}'. Use: auto, imagenet, zero_one, raw255.")
    return norm


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _convert_norm(images_nhwc: np.ndarray, src_norm: str, dst_norm: str) -> np.ndarray:
    """
    Convert NHWC float images between normalization modes.

    Supported norms:
    - imagenet: (img/255 - mean)/std
    - zero_one: img/255
    - raw255:   img
    - m1p1:     img/127.5 - 1
    """
    src = _normalize_input_norm(src_norm)
    dst = _normalize_input_norm(dst_norm)
    if src == dst:
        return images_nhwc

    x = images_nhwc.astype(np.float32, copy=False)

    # src -> raw255
    if src == "imagenet":
        x = (x * IMAGENET_STD[None, None, None, :]) + IMAGENET_MEAN[None, None, None, :]
        x = x * 255.0
    elif src == "zero_one":
        x = x * 255.0
    elif src == "raw255":
        pass
    elif src == "m1p1":
        x = (x + 1.0) * 127.5
    else:
        raise ValueError(f"Unsupported src_norm='{src_norm}'")

    # raw255 -> dst
    if dst == "imagenet":
        x = x / 255.0
        x = (x - IMAGENET_MEAN[None, None, None, :]) / IMAGENET_STD[None, None, None, :]
    elif dst == "zero_one":
        x = x / 255.0
    elif dst == "raw255":
        pass
    elif dst == "m1p1":
        x = (x / 127.5) - 1.0
    else:
        raise ValueError(f"Unsupported dst_norm='{dst_norm}'")

    return x


class OnnxHeatmapTeacher:
    """
    Adapter to use an ONNX heatmap teacher inside the existing distillation loop.

    Expected ONNX:
      input:  img [B,3,H,W] float32
      output: heatmap [B,4,h,w] float32 (typically after sigmoid)

    Output dict matches keys expected by StudentDistillTrainer:
      - simcc_x: [B,4,num_bins] float32 logits (from heatmap marginals)
      - simcc_y: [B,4,num_bins] float32 logits
      - coords:  [B,8] float32 in [0,1] (soft-argmax on heatmap marginals)
      - score_logit: [B,1] float32 (derived confidence; optional)
    """

    def __init__(
        self,
        onnx_path: str,
        num_bins: int,
        teacher_input_norm: str,
        dataset_input_norm: str,
        provider: str = "auto",
    ):
        self.onnx_path = str(onnx_path)
        self.num_bins = int(num_bins)
        self.teacher_input_norm = _normalize_input_norm(teacher_input_norm)
        self.dataset_input_norm = _normalize_input_norm(dataset_input_norm)

        try:
            import onnxruntime as ort
        except Exception as e:
            raise RuntimeError("onnxruntime is required for --teacher_onnx") from e

        if provider == "cpu":
            providers = ["CPUExecutionProvider"]
        elif provider == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.sess = ort.InferenceSession(self.onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def __call__(self, images, training: bool = False):
        x = images.numpy().astype(np.float32, copy=False)  # [B,H,W,3] in dataset_input_norm
        x = _convert_norm(x, src_norm=self.dataset_input_norm, dst_norm=self.teacher_input_norm)
        x = np.transpose(x, (0, 3, 1, 2))  # NCHW

        heat = self.sess.run([self.output_name], {self.input_name: x})[0]  # [B,4,h,w]
        heat = heat.astype(np.float32, copy=False)
        b, c, h, w = heat.shape
        if c != 4:
            raise RuntimeError(f"Unexpected teacher heatmap channels={c} (expected 4)")

        # Marginals: [B,4,w] and [B,4,h]
        x_m = heat.sum(axis=2)
        y_m = heat.sum(axis=3)

        # Upsample to num_bins via linear interpolation on index space.
        src_x = np.linspace(0.0, float(w - 1), w, dtype=np.float32)
        src_y = np.linspace(0.0, float(h - 1), h, dtype=np.float32)
        dst_x = np.linspace(0.0, float(w - 1), self.num_bins, dtype=np.float32)
        dst_y = np.linspace(0.0, float(h - 1), self.num_bins, dtype=np.float32)

        def interp_last(arr, src_grid, dst_grid):
            out = np.empty((arr.shape[0], arr.shape[1], len(dst_grid)), dtype=np.float32)
            for bi in range(arr.shape[0]):
                for ci in range(arr.shape[1]):
                    out[bi, ci] = np.interp(dst_grid, src_grid, arr[bi, ci]).astype(np.float32)
            return out

        x_up = interp_last(x_m, src_x, dst_x)
        y_up = interp_last(y_m, src_y, dst_y)

        eps = 1e-9
        simcc_x_logits = np.log(np.maximum(x_up, eps))
        simcc_y_logits = np.log(np.maximum(y_up, eps))

        # Soft-argmax coords (in [0,1]) from original marginals.
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        x_sum = x_m.sum(axis=-1) + eps
        y_sum = y_m.sum(axis=-1) + eps
        x_px = (x_m * xs[None, None, :]).sum(axis=-1) / x_sum
        y_px = (y_m * ys[None, None, :]).sum(axis=-1) / y_sum
        x01 = x_px / float(max(w - 1, 1))
        y01 = y_px / float(max(h - 1, 1))

        coords = np.stack([x01, y01], axis=-1).reshape(b, 8).astype(np.float32)
        coords = np.clip(coords, 0.0, 1.0)

        peak = np.max(heat, axis=(2, 3))  # [B,4]
        conf = np.clip(np.mean(peak, axis=1), 1e-6, 1.0 - 1e-6)  # [B]
        score_logit = np.log(conf / (1.0 - conf)).astype(np.float32)[:, None]  # [B,1]

        return {
            "simcc_x": tf.convert_to_tensor(simcc_x_logits),
            "simcc_y": tf.convert_to_tensor(simcc_y_logits),
            "coords": tf.convert_to_tensor(coords),
            "score_logit": tf.convert_to_tensor(score_logit),
        }


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


def _load_teacher_kwargs(args) -> tuple[dict, dict]:
    """Load teacher model kwargs from config.json if available."""
    teacher_weights_path = Path(args.teacher_weights)

    cfg = {}
    config_path = None
    if args.teacher_config:
        config_path = Path(args.teacher_config)
    else:
        candidate = teacher_weights_path.parent / "config.json"
        if candidate.exists():
            config_path = candidate

    teacher_kwargs = {
        "backbone": args.teacher_backbone,
        "backbone_minimalistic": args.teacher_backbone_minimalistic,
        "backbone_include_preprocessing": args.teacher_backbone_include_preprocessing,
        "alpha": args.teacher_alpha,
        "fpn_ch": args.teacher_fpn_ch,
        "simcc_ch": args.teacher_simcc_ch,
        "img_size": args.img_size,
        "num_bins": args.num_bins,
        "tau": args.teacher_tau,
    }

    if config_path and config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        teacher_kwargs["backbone"] = cfg.get("backbone", teacher_kwargs["backbone"])
        teacher_kwargs["backbone_minimalistic"] = cfg.get(
            "backbone_minimalistic", teacher_kwargs["backbone_minimalistic"]
        )
        teacher_kwargs["backbone_include_preprocessing"] = cfg.get(
            "backbone_include_preprocessing", teacher_kwargs["backbone_include_preprocessing"]
        )
        teacher_kwargs["alpha"] = cfg.get("alpha", teacher_kwargs["alpha"])
        teacher_kwargs["fpn_ch"] = cfg.get("fpn_ch", teacher_kwargs["fpn_ch"])
        teacher_kwargs["simcc_ch"] = cfg.get("simcc_ch", teacher_kwargs["simcc_ch"])
        teacher_kwargs["img_size"] = cfg.get("img_size", teacher_kwargs["img_size"])
        teacher_kwargs["num_bins"] = cfg.get("num_bins", teacher_kwargs["num_bins"])
        teacher_kwargs["tau"] = cfg.get("tau", teacher_kwargs["tau"])
        print(f"Loaded teacher config from {config_path}")

    # Enforce shape compatibility
    if teacher_kwargs["img_size"] != args.img_size:
        raise ValueError(
            f"Teacher img_size={teacher_kwargs['img_size']} != --img_size={args.img_size}. "
            "Use matching shapes for distillation."
        )
    if teacher_kwargs["num_bins"] != args.num_bins:
        raise ValueError(
            f"Teacher num_bins={teacher_kwargs['num_bins']} != --num_bins={args.num_bins}. "
            "Use matching bins for distillation."
        )

    return teacher_kwargs, cfg


def _find_split_file(data_root: Path, split: str) -> Path:
    """Match the same split-file fallback logic as dataset.create_dataset()."""
    for prefix in (f"{split}_with_negative_v2", f"{split}_with_negative", split):
        candidate = data_root / f"{prefix}.txt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No split file found for split='{split}' in {data_root}")


class StudentDistillTrainer:
    """Custom trainer for distillation with progress bars."""

    def __init__(
        self,
        student: keras.Model,
        teacher: keras.Model,
        optimizer: keras.optimizers.Optimizer,
        img_size: int = 224,
        sigma_px: float = 3.0,
        tau: float = 1.0,
        distill_tau: float = 2.0,
        w_simcc: float = 1.0,
        w_coord: float = 0.5,
        w_score: float = 0.5,
        w_distill_simcc: float = 1.0,
        w_distill_coord: float = 0.0,
        w_distill_score: float = 0.1,
        fast_mode: bool = False,
        augment: bool = False,
        image_norm: str = "imagenet",
    ):
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.img_size = img_size
        self.sigma_px = sigma_px
        self.tau = tau
        self.distill_tau = distill_tau
        self.w_simcc = w_simcc
        self.w_coord = w_coord
        self.w_score = w_score
        self.w_distill_simcc = w_distill_simcc
        self.w_distill_coord = w_distill_coord
        self.w_distill_score = w_distill_score
        self.fast_mode = fast_mode
        self.augment = bool(augment)
        self.image_norm = str(image_norm)

        # Trackers
        self.train_loss = keras.metrics.Mean(name="loss")
        self.train_loss_hard_simcc = keras.metrics.Mean(name="loss_hard_simcc")
        self.train_loss_hard_coord = keras.metrics.Mean(name="loss_hard_coord")
        self.train_loss_hard_score = keras.metrics.Mean(name="loss_hard_score")
        self.train_loss_distill_simcc = keras.metrics.Mean(name="loss_distill_simcc")
        self.train_loss_distill_coord = keras.metrics.Mean(name="loss_distill_coord")
        self.train_loss_distill_score = keras.metrics.Mean(name="loss_distill_score")
        self.train_iou = keras.metrics.Mean(name="iou")
        self.train_corner_err = keras.metrics.Mean(name="corner_err")

        self.val_loss = keras.metrics.Mean(name="val_loss")
        self.val_loss_hard_simcc = keras.metrics.Mean(name="val_loss_hard_simcc")
        self.val_loss_hard_coord = keras.metrics.Mean(name="val_loss_hard_coord")
        self.val_loss_hard_score = keras.metrics.Mean(name="val_loss_hard_score")

    def _compute_hard_losses(self, outputs, targets):
        simcc_x = outputs["simcc_x"]
        simcc_y = outputs["simcc_y"]
        score_logit = outputs["score_logit"]
        coords_pred = outputs["coords"]

        has_doc = tf.cast(targets["has_doc"], tf.float32)
        if len(has_doc.shape) == 2:
            has_doc = tf.squeeze(has_doc, axis=-1)
        coords_gt = tf.cast(targets["coords"], tf.float32)

        # SimCC hard targets (only positive)
        gt_coords_4x2 = tf.reshape(coords_gt, [-1, 4, 2])
        gt_x = gt_coords_4x2[:, :, 0]
        gt_y = gt_coords_4x2[:, :, 1]

        target_x = gaussian_1d_targets(gt_x, self.img_size, self.sigma_px)
        target_y = gaussian_1d_targets(gt_y, self.img_size, self.sigma_px)

        log_pred_x = tf.nn.log_softmax(simcc_x / self.tau, axis=-1)
        log_pred_y = tf.nn.log_softmax(simcc_y / self.tau, axis=-1)

        ce_x = -tf.reduce_sum(target_x * log_pred_x, axis=-1)  # [B, 4]
        ce_y = -tf.reduce_sum(target_y * log_pred_y, axis=-1)
        ce = tf.reduce_mean(ce_x + ce_y, axis=-1)  # [B]

        loss_hard_simcc = tf.reduce_sum(ce * has_doc) / (tf.reduce_sum(has_doc) + 1e-9)

        # Coord hard loss (only positive)
        loss_per_coord = tf.abs(coords_pred - coords_gt)
        loss_per_sample = tf.reduce_mean(loss_per_coord, axis=-1)
        loss_hard_coord = tf.reduce_sum(loss_per_sample * has_doc) / (tf.reduce_sum(has_doc) + 1e-9)

        # Score hard loss (all samples)
        loss_hard_score = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=has_doc[:, None],
            logits=score_logit,
        )
        loss_hard_score = tf.reduce_mean(loss_hard_score)

        return {
            "hard_simcc": loss_hard_simcc,
            "hard_coord": loss_hard_coord,
            "hard_score": loss_hard_score,
        }

    def _compute_distill_losses(self, student_outputs, teacher_outputs, targets):
        has_doc = tf.cast(targets["has_doc"], tf.float32)
        if len(has_doc.shape) == 2:
            has_doc = tf.squeeze(has_doc, axis=-1)

        # SimCC distillation (only positive samples)
        # Use teacher softmax as targets, student log-softmax as predictions.
        temperature = tf.cast(self.distill_tau, tf.float32)
        eps = tf.constant(1e-9, tf.float32)

        loss_distill_simcc = tf.constant(0.0, tf.float32)
        if self.w_distill_simcc > 0:
            for key in ("simcc_x", "simcc_y"):
                t_logits = tf.cast(teacher_outputs[key], tf.float32)
                s_logits = tf.cast(student_outputs[key], tf.float32)

                t_prob = tf.nn.softmax(t_logits / temperature, axis=-1)
                s_log_prob = tf.nn.log_softmax(s_logits / temperature, axis=-1)

                # Cross-entropy(t_prob, s_prob) = -sum(t_prob * log s_prob)
                ce = -tf.reduce_sum(t_prob * s_log_prob, axis=-1)  # [B, 4]
                ce = tf.reduce_mean(ce, axis=-1)  # [B]
                ce = tf.reduce_sum(ce * has_doc) / (tf.reduce_sum(has_doc) + eps)
                loss_distill_simcc += ce

            # Scale as in standard distillation
            loss_distill_simcc = loss_distill_simcc * (temperature * temperature)

        # Coord distillation (only positive)
        loss_distill_coord = tf.constant(0.0, tf.float32)
        if self.w_distill_coord > 0:
            t_coords = tf.cast(teacher_outputs["coords"], tf.float32)
            s_coords = tf.cast(student_outputs["coords"], tf.float32)
            diff = tf.abs(s_coords - t_coords)
            per_sample = tf.reduce_mean(diff, axis=-1)
            loss_distill_coord = tf.reduce_sum(per_sample * has_doc) / (tf.reduce_sum(has_doc) + eps)

        # Score distillation (all samples)
        loss_distill_score = tf.constant(0.0, tf.float32)
        if self.w_distill_score > 0:
            t_score = tf.cast(teacher_outputs["score_logit"], tf.float32)
            s_score = tf.cast(student_outputs["score_logit"], tf.float32)
            loss_distill_score = tf.reduce_mean(tf.square(s_score - t_score))

        return {
            "distill_simcc": loss_distill_simcc,
            "distill_coord": loss_distill_coord,
            "distill_score": loss_distill_score,
        }

    def _compute_metrics(self, coords_pred, coords_gt, has_doc):
        """Compute coarse IoU + corner error for positive samples (fast)."""
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
    def train_step(self, images, targets, apply_augment: bool = False):
        coords = targets["coords"]
        has_doc = targets["has_doc"]
        # is_outlier may or may not be present depending on dataset mode
        is_outlier = targets["is_outlier"] if "is_outlier" in targets else None

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

        teacher_outputs = self.teacher(images, training=False)
        teacher_outputs = {k: tf.stop_gradient(v) for k, v in teacher_outputs.items()}

        with tf.GradientTape() as tape:
            student_outputs = self.student(images, training=True)

            hard = self._compute_hard_losses(student_outputs, targets)
            distill = self._compute_distill_losses(student_outputs, teacher_outputs, targets)

            total_loss = (
                self.w_simcc * hard["hard_simcc"]
                + self.w_coord * hard["hard_coord"]
                + self.w_score * hard["hard_score"]
                + self.w_distill_simcc * distill["distill_simcc"]
                + self.w_distill_coord * distill["distill_coord"]
                + self.w_distill_score * distill["distill_score"]
            )

        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        has_doc_1d = tf.cast(targets["has_doc"], tf.float32)
        if len(has_doc_1d.shape) == 2:
            has_doc_1d = tf.squeeze(has_doc_1d, axis=-1)
        iou, corner_err = self._compute_metrics(student_outputs["coords"], targets["coords"], has_doc_1d)

        losses = {"total": total_loss, **hard, **distill}
        return losses, iou, corner_err

    def train_epoch(self, train_ds, num_batches: int):
        for metric in (
            self.train_loss,
            self.train_loss_hard_simcc,
            self.train_loss_hard_coord,
            self.train_loss_hard_score,
            self.train_loss_distill_simcc,
            self.train_loss_distill_coord,
            self.train_loss_distill_score,
            self.train_iou,
            self.train_corner_err,
        ):
            metric.reset_state()

        pbar = tqdm(
            train_ds,
            desc="  Train",
            total=num_batches,
            unit="batch",
            ncols=100,
            leave=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]",
            ascii="░█",
        )

        for images, targets in pbar:
            losses, iou, corner_err = self.train_step(
                images,
                targets,
                apply_augment=(self.fast_mode and self.augment),
            )

            self.train_loss.update_state(losses["total"])
            self.train_loss_hard_simcc.update_state(losses["hard_simcc"])
            self.train_loss_hard_coord.update_state(losses["hard_coord"])
            self.train_loss_hard_score.update_state(losses["hard_score"])
            self.train_loss_distill_simcc.update_state(losses["distill_simcc"])
            self.train_loss_distill_coord.update_state(losses["distill_coord"])
            self.train_loss_distill_score.update_state(losses["distill_score"])

            self.train_iou.update_state(iou)
            self.train_corner_err.update_state(corner_err)

            pbar.set_postfix(
                {
                    "loss": f"{float(self.train_loss.result()):.4f}",
                    "iou": f"{float(self.train_iou.result()):.3f}",
                    "err": f"{float(self.train_corner_err.result()):.1f}px",
                }
            )

        return {
            "loss": float(self.train_loss.result()),
            "loss_hard_simcc": float(self.train_loss_hard_simcc.result()),
            "loss_hard_coord": float(self.train_loss_hard_coord.result()),
            "loss_hard_score": float(self.train_loss_hard_score.result()),
            "loss_distill_simcc": float(self.train_loss_distill_simcc.result()),
            "loss_distill_coord": float(self.train_loss_distill_coord.result()),
            "loss_distill_score": float(self.train_loss_distill_score.result()),
            "iou": float(self.train_iou.result()),
            "corner_err_px": float(self.train_corner_err.result()),
        }

    def validate(self, val_ds, num_batches: int):
        for metric in (
            self.val_loss,
            self.val_loss_hard_simcc,
            self.val_loss_hard_coord,
            self.val_loss_hard_score,
        ):
            metric.reset_state()

        metrics = ValidationMetrics(img_size=self.img_size)

        pbar = tqdm(
            val_ds,
            desc="  Val  ",
            total=num_batches,
            unit="batch",
            ncols=100,
            leave=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]",
            ascii="░█",
        )

        for images, targets in pbar:
            outputs = self.student(images, training=False)
            hard = self._compute_hard_losses(outputs, targets)
            total = self.w_simcc * hard["hard_simcc"] + self.w_coord * hard["hard_coord"] + self.w_score * hard["hard_score"]

            self.val_loss.update_state(total)
            self.val_loss_hard_simcc.update_state(hard["hard_simcc"])
            self.val_loss_hard_coord.update_state(hard["hard_coord"])
            self.val_loss_hard_score.update_state(hard["hard_score"])

            coords_pred = outputs["coords"].numpy()
            score_pred = tf.sigmoid(outputs["score_logit"]).numpy()
            coords_gt = targets["coords"].numpy()
            has_doc = targets["has_doc"].numpy()
            metrics.update(coords_pred, coords_gt, score_pred, has_doc)

            pbar.set_postfix({"loss": f"{float(self.val_loss.result()):.4f}"})

        results = metrics.compute()

        return {
            "loss": float(self.val_loss.result()),
            "loss_hard_simcc": float(self.val_loss_hard_simcc.result()),
            "loss_hard_coord": float(self.val_loss_hard_coord.result()),
            "loss_hard_score": float(self.val_loss_hard_score.result()),
            **results,
        }


def main():
    args = parse_args()

    print_device_info()

    teacher_cfg = {}
    if args.teacher_onnx:
        # Keep student normalization explicit; default to imagenet (matches our TF models).
        args.input_norm = "imagenet" if _normalize_input_norm(args.input_norm) == "auto" else _normalize_input_norm(args.input_norm)
        teacher_kwargs = {"img_size": args.img_size, "num_bins": args.num_bins}
        teacher_cfg = {"type": "onnx_heatmap", "input_norm": args.teacher_onnx_input_norm}
    else:
        teacher_kwargs, teacher_cfg = _load_teacher_kwargs(args)
        resolved_input_norm = _resolve_input_norm(
            args.input_norm, teacher_cfg, teacher_kwargs["backbone_include_preprocessing"]
        )
        if args.input_norm != "auto" and teacher_cfg.get("input_norm"):
            teacher_norm = _normalize_input_norm(teacher_cfg.get("input_norm"))
            if teacher_norm != "auto" and teacher_norm != resolved_input_norm:
                print(
                    f"Warning: --input_norm={resolved_input_norm} differs from teacher config input_norm={teacher_cfg.get('input_norm')}"
                )
        args.input_norm = resolved_input_norm

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DocCornerNetV3 Student Distillation")
    print("=" * 80)
    if args.teacher_onnx:
        print(f"Teacher ONNX: {args.teacher_onnx}")
        print(f"Teacher ONNX input_norm: {args.teacher_onnx_input_norm}")
    else:
        print(f"Teacher weights: {args.teacher_weights}")
    print(f"Output: {output_dir}")

    # Save config
    config = vars(args)
    config["teacher_resolved"] = teacher_kwargs
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Count samples (same naming scheme as train.py)
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
    augment_config = {
        "rotation_degrees": 10,
        "scale_range": (0.8, 1.1),
        "brightness": 0.25,
        "contrast": 0.25,
        "saturation": 0.15,
        "blur_prob": 0.2,
        "blur_kernel": 3,
        "horizontal_flip": True,
    }
    augment_config_outlier = {
        "rotation_degrees": 15,
        "scale_range": (0.75, 1.15),
        "brightness": 0.3,
        "contrast": 0.3,
        "saturation": 0.2,
        "blur_prob": 0.25,
        "blur_kernel": 5,
        "horizontal_flip": True,
    }

    shared_cache = None
    if args.cache_images:
        data_root_path = Path(args.data_root)
        image_dir = data_root_path / "images"
        negative_dir = data_root_path / args.negative_dir

        all_images = []
        for split_name in [args.train_split, args.val_split]:
            for prefix in [f"{split_name}_with_negative_v2", f"{split_name}_with_negative", split_name]:
                candidate = data_root_path / f"{prefix}.txt"
                if candidate.exists():
                    all_images.extend(load_split_file(str(candidate)))
                    break

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
        augment=args.augment if not use_fast_mode else False,
        augment_config=augment_config if args.augment and not use_fast_mode else None,
        augment_config_outlier=augment_config_outlier if args.augment and not use_fast_mode else None,
        outlier_list=args.outlier_list,  # Pass outlier_list for both modes (fast_mode uses it for is_outlier flag)
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

    # Teacher
    print("\nLoading teacher...")
    if args.teacher_onnx:
        teacher = OnnxHeatmapTeacher(
            onnx_path=args.teacher_onnx,
            num_bins=args.num_bins,
            teacher_input_norm=args.teacher_onnx_input_norm,
            dataset_input_norm=args.input_norm,
            provider=args.teacher_onnx_provider,
        )
        print("Teacher type: ONNX heatmap")
    else:
        teacher = create_model(
            backbone=teacher_kwargs["backbone"],
            alpha=teacher_kwargs["alpha"],
            backbone_minimalistic=teacher_kwargs["backbone_minimalistic"],
            backbone_include_preprocessing=teacher_kwargs["backbone_include_preprocessing"],
            fpn_ch=teacher_kwargs["fpn_ch"],
            simcc_ch=teacher_kwargs["simcc_ch"],
            img_size=teacher_kwargs["img_size"],
            num_bins=teacher_kwargs["num_bins"],
            tau=teacher_kwargs["tau"],
            backbone_weights=_normalize_backbone_weights(args.teacher_backbone_weights),
        )
        teacher.load_weights(args.teacher_weights)
        teacher.trainable = False
        print(f"Teacher parameters: {teacher.count_params():,}")

    # Student
    print("\nCreating student...")
    student = create_model(
        backbone=args.student_backbone,
        alpha=args.student_alpha,
        backbone_minimalistic=args.student_backbone_minimalistic,
        backbone_include_preprocessing=args.student_backbone_include_preprocessing,
        fpn_ch=args.student_fpn_ch,
        simcc_ch=args.student_simcc_ch,
        img_size=args.img_size,
        num_bins=args.num_bins,
        tau=args.tau,
        backbone_weights=_normalize_backbone_weights(args.student_backbone_weights),
    )
    print(f"Student parameters: {student.count_params():,}")

    if args.student_init_weights:
        init_path = _find_init_weights_path(args.student_init_weights)
        print(f"\nLoading student init weights from: {init_path}")
        try:
            student.load_weights(str(init_path))
            print("✓ Loaded student init weights (strict)")
        except Exception as e:
            if not args.student_init_partial:
                raise
            print(f"Warning: strict student init load failed: {e}")
            print("Retrying with by_name=True, skip_mismatch=True...")
            student.load_weights(str(init_path), by_name=True, skip_mismatch=True)
            print("✓ Loaded student init weights (partial)")

    optimizer = keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    trainer = StudentDistillTrainer(
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        img_size=args.img_size,
        sigma_px=args.sigma_px,
        tau=args.tau,
        distill_tau=args.distill_tau,
        w_simcc=args.w_simcc,
        w_coord=args.w_coord,
        w_score=args.w_score,
        w_distill_simcc=args.w_distill_simcc,
        w_distill_coord=args.w_distill_coord,
        w_distill_score=args.w_distill_score,
        fast_mode=use_fast_mode,
        augment=args.augment,
        image_norm=args.input_norm,
    )

    best_iou = 0.0
    best_epoch = 0
    current_lr = args.lr
    no_improve_count = 0
    lr_no_improve_count = 0
    history = {"train": [], "val": []}

    print(f"\n{'=' * 80}")
    print(f"Starting student training for {args.epochs} epochs...")
    print(f"{'=' * 80}")

    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            optimizer.learning_rate.assign(warmup_lr)
            current_lr = warmup_lr

        print(f"\n{'─' * 80}")
        print(f"Epoch {epoch + 1}/{args.epochs}  (LR: {current_lr:.2e})")
        print(f"{'─' * 80}")

        train_metrics = trainer.train_epoch(train_ds, num_train_batches)
        val_metrics = trainer.validate(val_ds, num_val_batches)

        print("\n  Summary:")
        print(
            f"    Train: loss={train_metrics['loss']:.4f} "
            f"(hard_simcc={train_metrics['loss_hard_simcc']:.4f} "
            f"hard_coord={train_metrics['loss_hard_coord']:.4f} "
            f"hard_score={train_metrics['loss_hard_score']:.4f} "
            f"dist_simcc={train_metrics['loss_distill_simcc']:.4f})"
        )
        print(f"           err={train_metrics['corner_err_px']:.1f}px  IoU={train_metrics['iou']:.3f}")
        print(
            f"    Val:   loss={val_metrics['loss']:.4f} "
            f"(hard_simcc={val_metrics['loss_hard_simcc']:.4f} "
            f"hard_coord={val_metrics['loss_hard_coord']:.4f} "
            f"hard_score={val_metrics['loss_hard_score']:.4f})"
        )
        print(
            f"           err_mean={val_metrics['corner_error_px']:.1f}px  "
            f"err_p95={val_metrics['corner_error_p95_px']:.1f}px"
        )
        print(
            f"           IoU={val_metrics['mean_iou']:.4f}  "
            f"R@90={val_metrics['recall_90']*100:.1f}%  "
            f"R@95={val_metrics['recall_95']*100:.1f}%"
        )
        print(f"           cls_acc={val_metrics['cls_accuracy']*100:.1f}%  cls_f1={val_metrics['cls_f1']:.3f}")

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if val_metrics["mean_iou"] > best_iou:
            best_iou = val_metrics["mean_iou"]
            best_epoch = epoch + 1
            no_improve_count = 0
            lr_no_improve_count = 0

            student.save(output_dir / "best_student.keras")
            student.save_weights(str(output_dir / "best_student.weights.h5"))
            inference_student = create_inference_model(student)
            inference_student.save(output_dir / "best_student_inference.keras")
            print(f"  ★ Saved best student (IoU: {best_iou:.4f})")
        else:
            no_improve_count += 1
            lr_no_improve_count += 1

            if epoch >= args.warmup_epochs:
                if lr_no_improve_count >= args.lr_patience and current_lr > args.min_lr:
                    current_lr = max(current_lr * args.lr_factor, args.min_lr)
                    optimizer.learning_rate.assign(current_lr)
                    lr_no_improve_count = 0
                    print(f"  ↓ Reduced LR to {current_lr:.2e}")

        if no_improve_count >= args.patience:
            print(
                f"\n⚠ Early stopping at epoch {epoch + 1} "
                f"(no improvement for {args.patience} epochs)"
            )
            break

    # Save final
    student.save(output_dir / "final_student.keras")
    student.save_weights(str(output_dir / "final_student.weights.h5"))
    inference_student = create_inference_model(student)
    inference_student.save(output_dir / "final_student_inference.keras")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'=' * 80}")
    print("Student Training Complete!")
    print(f"{'=' * 80}")
    print(f"Best epoch: {best_epoch} with IoU: {best_iou:.4f}")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
