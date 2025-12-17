"""
Evaluation script for DocCornerNetV3 (TensorFlow/Keras).

Evaluates a trained model and reports detailed metrics.
"""

import argparse
import json
from pathlib import Path

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
    parser.add_argument("--data_root", type=str,
                        default="../../datasets/official/doc-scanner-dataset-labeled",
                        help="Path to dataset root")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")

    # Model config (if loading weights only)
    parser.add_argument("--alpha", type=float, default=0.75,
                        help="MobileNetV3 width multiplier (0.75 or 1.0)")
    parser.add_argument("--fpn_ch", type=int, default=48,
                        help="FPN channels")
    parser.add_argument("--dec_ch", type=int, default=32,
                        help="Decoder channels")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Softmax temperature")

    return parser.parse_args()


def load_model(args):
    """Load model from path (SavedModel or weights)."""
    model_path = Path(args.model_path)

    # Try to load config if available
    config_path = model_path.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        alpha = config.get("alpha", args.alpha)
        fpn_ch = config.get("fpn_ch", args.fpn_ch)
        dec_ch = config.get("dec_ch", args.dec_ch)
        img_size = config.get("img_size", args.img_size)
        tau = config.get("tau", args.tau)
        print(f"Loaded config from {config_path}")
    else:
        alpha = args.alpha
        fpn_ch = args.fpn_ch
        dec_ch = args.dec_ch
        img_size = args.img_size
        tau = args.tau

    # Create model
    model = create_model(
        alpha=alpha,
        fpn_ch=fpn_ch,
        dec_ch=dec_ch,
        img_size=img_size,
        tau=tau,
    )

    # Load weights
    if model_path.suffix == ".h5":
        model.load_weights(str(model_path))
        print(f"Loaded weights from {model_path}")
    elif model_path.is_dir():
        # Try to load as SavedModel
        try:
            model = keras.models.load_model(str(model_path))
            print(f"Loaded SavedModel from {model_path}")
        except:
            # Try to find weights file
            for weights_file in ["best_iou_weights.h5", "final_weights.h5", "latest_weights.h5"]:
                weights_path = model_path / weights_file
                if weights_path.exists():
                    model.load_weights(str(weights_path))
                    print(f"Loaded weights from {weights_path}")
                    break
    else:
        raise ValueError(f"Cannot load model from {model_path}")

    return model, img_size


def main():
    args = parse_args()

    print("=" * 60)
    print("DocCornerNetV3 Evaluation")
    print("=" * 60)

    # Load model
    model, img_size = load_model(args)
    print(f"Model parameters: {model.count_params():,}")

    # Create dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = create_dataset(
        data_root=args.data_root,
        split=args.split,
        img_size=img_size,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
    )

    # Evaluate
    print(f"\nEvaluating...")
    metrics = ValidationMetrics(img_size=img_size)

    for images, targets in tqdm(dataset, desc="Evaluating"):
        outputs = model(images, training=False)

        coords_pred = outputs["coords"].numpy()
        score_logit = outputs["score_logit"].numpy()
        score_pred = 1.0 / (1.0 + np.exp(-score_logit))  # sigmoid

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
