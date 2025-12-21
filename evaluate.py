"""
Evaluation script for DocCornerNetV3 (TensorFlow/Keras).

Evaluates a trained model and reports detailed metrics.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from model import create_model
from dataset import create_dataset
from metrics import ValidationMetrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DocCornerNetV3 model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved model directory or weights file")
    parser.add_argument(
        "--unsafe_load",
        action="store_true",
        help=(
            "Allow unsafe Keras deserialization (e.g., Lambda layers) when loading a serialized .keras model. "
            "Use only for artifacts you trust."
        ),
    )
    parser.add_argument("--data_root", type=str,
                        default="../../datasets/official/doc-scanner-dataset-labeled",
                        help="Path to dataset root")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split name to evaluate (supports custom split files like val_cleaned)",
    )
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument(
        "--input_norm",
        type=str,
        default="imagenet",
        choices=["imagenet", "zero_one", "raw255"],
        help="Input normalization for dataset images (must match how the checkpoint was trained)",
    )

    # Model config (if loading weights only)
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
        default=None,
        help="Backbone init weights ('imagenet' or None). None avoids downloads.",
    )
    parser.add_argument("--fpn_ch", type=int, default=48,
                        help="FPN channels")
    parser.add_argument("--simcc_ch", type=int, default=128,
                        help="SimCC head channels")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--num_bins", type=int, default=224,
                        help="Number of bins for coordinate classification")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Softmax temperature")

    return parser.parse_args()


def _normalize_backbone_weights(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None
    return value


def _find_config_path(model_path: Path) -> Optional[Path]:
    candidates = []
    if model_path.is_dir():
        candidates.extend([model_path / "config.json", model_path.parent / "config.json"])
    else:
        candidates.extend([model_path.parent / "config.json"])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_model(args):
    """Load model from path (SavedModel or weights)."""
    model_path = Path(args.model_path)

    # Try to load config if available (to reconstruct architecture for weights-only)
    config_path = _find_config_path(model_path)
    if config_path and config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

        # Support distillation/student configs (train_student.py) which store student_* keys.
        is_student = any(
            k in config
            for k in (
                "student_backbone",
                "student_alpha",
                "student_fpn_ch",
                "student_simcc_ch",
                "student_backbone_minimalistic",
                "student_backbone_include_preprocessing",
            )
        )
        if is_student:
            backbone = config.get("student_backbone", config.get("backbone", args.backbone))
            alpha = config.get("student_alpha", config.get("alpha", args.alpha))
            fpn_ch = config.get("student_fpn_ch", config.get("fpn_ch", args.fpn_ch))
            simcc_ch = config.get("student_simcc_ch", config.get("simcc_ch", args.simcc_ch))
            backbone_minimalistic = args.backbone_minimalistic or config.get(
                "student_backbone_minimalistic", config.get("backbone_minimalistic", False)
            )
            if args.backbone_include_preprocessing:
                backbone_include_preprocessing = True
            else:
                backbone_include_preprocessing = bool(
                    config.get(
                        "student_backbone_include_preprocessing",
                        config.get("backbone_include_preprocessing", False),
                    )
                )
        else:
            backbone = config.get("backbone", args.backbone)
            alpha = config.get("alpha", args.alpha)
            fpn_ch = config.get("fpn_ch", args.fpn_ch)
            simcc_ch = config.get("simcc_ch", args.simcc_ch)
            backbone_minimalistic = args.backbone_minimalistic or config.get(
                "backbone_minimalistic", False
            )
            # CLI should be able to override config for backwards-compatible eval.
            if args.backbone_include_preprocessing:
                backbone_include_preprocessing = True
            else:
                backbone_include_preprocessing = bool(config.get("backbone_include_preprocessing", False))

        # CLI should be able to override config for backwards-compatible eval.
        img_size = config.get("img_size", args.img_size)
        num_bins = config.get("num_bins", args.num_bins)
        tau = config.get("tau", args.tau)
        print(f"Loaded config from {config_path}")
    else:
        backbone = args.backbone
        backbone_minimalistic = args.backbone_minimalistic
        backbone_include_preprocessing = args.backbone_include_preprocessing
        alpha = args.alpha
        fpn_ch = args.fpn_ch
        simcc_ch = args.simcc_ch
        img_size = args.img_size
        num_bins = args.num_bins
        tau = args.tau

    # Load serialized model if possible (.keras or SavedModel directory)
    def _try_load_serialized(p: Path):
        try:
            if args.unsafe_load:
                try:
                    keras.config.enable_unsafe_deserialization()
                except Exception:
                    pass
            return keras.models.load_model(str(p), compile=False, safe_mode=not args.unsafe_load)
        except TypeError:
            # Older TF/Keras versions don't support safe_mode.
            return keras.models.load_model(str(p), compile=False)

    if model_path.is_dir() or model_path.suffix == ".keras":
        try:
            model = _try_load_serialized(model_path)
            print(f"Loaded model from {model_path}")
            try:
                inferred = model.input_shape[1]
                if isinstance(inferred, int):
                    img_size = inferred
            except Exception:
                pass
            return model, img_size, backbone_include_preprocessing
        except Exception as e:
            print(f"Warning: failed to load serialized model from {model_path}: {e}")

    # Otherwise: create model and load weights
    model = create_model(
        backbone=backbone,
        alpha=alpha,
        backbone_minimalistic=backbone_minimalistic,
        backbone_include_preprocessing=backbone_include_preprocessing,
        backbone_weights=_normalize_backbone_weights(args.backbone_weights),
        fpn_ch=fpn_ch,
        simcc_ch=simcc_ch,
        img_size=img_size,
        num_bins=num_bins,
        tau=tau,
    )

    # Load weights
    if model_path.suffix == ".h5":
        model.load_weights(str(model_path))
        print(f"Loaded weights from {model_path}")
    elif model_path.is_dir() or model_path.suffix == ".keras":
        # If we couldn't load a serialized .keras, fall back to nearby weights.
        # For directories, search within the directory. For .keras files, search next to the file.
        search_dir = model_path if model_path.is_dir() else model_path.parent
        preferred = []
        if model_path.suffix == ".keras":
            # Common convention: best_model.keras <-> best_model.weights.h5
            preferred.append(f"{model_path.stem}.weights.h5")
            preferred.append(f"{model_path.stem.replace('_inference', '')}.weights.h5")

        # Try to find weights file in directory
        for weights_file in [
            *preferred,
            "best_model.weights.h5",
            "final_model.weights.h5",
            "latest_weights.h5",
            "best_iou_weights.h5",
            "final_weights.h5",
            # Student/distillation artifacts
            "best_student.weights.h5",
            "final_student.weights.h5",
        ]:
            weights_path = search_dir / weights_file
            if weights_path.exists():
                model.load_weights(str(weights_path))
                print(f"Loaded weights from {weights_path}")
                break
        else:
            raise ValueError(f"Cannot find weights in {search_dir}")
    else:
        raise ValueError(f"Cannot load model from {model_path}")

    return model, img_size, backbone_include_preprocessing


def main():
    args = parse_args()

    print("=" * 60)
    print("DocCornerNetV3 Evaluation")
    print("=" * 60)

    # Load model
    model, img_size, backbone_include_preprocessing = load_model(args)
    print(f"Model parameters: {model.count_params():,}")

    # Create dataset
    print(f"\nLoading {args.split} dataset...")
    input_norm = args.input_norm
    print(f"Input normalization: {input_norm}")
    dataset = create_dataset(
        data_root=args.data_root,
        split=args.split,
        img_size=img_size,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
        drop_remainder=False,
        image_norm=input_norm,
    )

    # Evaluate
    print(f"\nEvaluating...")
    metrics = ValidationMetrics(img_size=img_size)

    for images, targets in tqdm(dataset, desc="Evaluating"):
        outputs = model(images, training=False)

        # Support both training model (dict outputs) and inference model (tuple/list outputs)
        if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
            coords_pred = outputs[0].numpy()
            score_logit = outputs[1].numpy()
        else:
            coords_pred = outputs["coords"].numpy()
            score_logit = outputs["score_logit"].numpy()
        # Numerically-stable sigmoid (avoid overflow warnings).
        score_pred = 1.0 / (1.0 + np.exp(-np.clip(score_logit, -60.0, 60.0)))

        coords_gt = targets["coords"].numpy()
        has_doc = targets["has_doc"].numpy()

        metrics.update(coords_pred, coords_gt, score_pred, has_doc)

    results = metrics.compute()

    # Print results
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS ({args.split} split)")
    print("=" * 60)

    print(f"\nGeometry Metrics (on {results['num_with_doc']} images with documents):")
    print(f"  Mean IoU:             {results['mean_iou']:.4f}")
    print(f"  Median IoU:           {results['median_iou']:.4f}")
    print()
    print(f"  Corner Error (mean):  {results['corner_error_px']:.2f} px")
    print(f"  Corner Error (p95):   {results['corner_error_p95_px']:.2f} px")
    print(f"  Corner Error (max):   {results['corner_error_max_px']:.2f} px")
    print()
    print(f"  Recall@50:            {results['recall_50']*100:.1f}%")
    print(f"  Recall@75:            {results['recall_75']*100:.1f}%")
    print(f"  Recall@90:            {results['recall_90']*100:.1f}%")
    print(f"  Recall@95:            {results['recall_95']*100:.1f}%")

    print(f"\nClassification Metrics (on {results['num_samples']} total images):")
    print(f"  Accuracy:             {results['cls_accuracy']*100:.1f}%")
    print(f"  Precision:            {results['cls_precision']*100:.1f}%")
    print(f"  Recall:               {results['cls_recall']*100:.1f}%")
    print(f"  F1 Score:             {results['cls_f1']*100:.1f}%")

    # Target comparison
    print("\n" + "-" * 60)
    print("Target Comparison:")

    iou = results['mean_iou']
    ce_mean = results['corner_error_px']
    ce_p95 = results['corner_error_p95_px']

    iou_target = 0.99
    ce_target = 1.0  # px

    iou_ok = "✓" if iou >= iou_target else "✗"
    ce_ok = "✓" if ce_mean <= ce_target else "✗"

    print(f"  IoU >= {iou_target*100:.0f}%:              {iou*100:.2f}%  {iou_ok}")
    print(f"  Corner Error <= {ce_target}px:     {ce_mean:.2f}px  {ce_ok}")

    print()


if __name__ == "__main__":
    main()
