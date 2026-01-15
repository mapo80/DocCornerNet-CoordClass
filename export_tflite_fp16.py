#!/usr/bin/env python3
"""
Export DocCornerNetV3 to TFLite FP16 format with GAU support.

This script handles models with GAU (Gated Attention Unit) which export.py doesn't support.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


def parse_args():
    p = argparse.ArgumentParser(
        description="Export DocCornerNetV3 to TFLite FP16",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (must contain config.json and best_model.weights.h5)",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output TFLite file path",
    )
    p.add_argument(
        "--output_format",
        type=str,
        default="coords9",
        choices=["coords9", "coords8"],
        help="Output format: coords9 (8 coords + 1 score) or coords8 (8 coords only)",
    )
    return p.parse_args()


def find_checkpoint_files(checkpoint_path: Path) -> tuple[Path, Path]:
    """Find config.json and weights file in checkpoint directory."""
    # Handle glob patterns (e.g., mobilenetv2_256_*)
    if "*" in str(checkpoint_path):
        import glob
        matches = sorted(glob.glob(str(checkpoint_path)))
        if not matches:
            raise FileNotFoundError(f"No matches for pattern: {checkpoint_path}")
        checkpoint_path = Path(matches[-1])  # Use latest

    # Find config.json
    config_candidates = [
        checkpoint_path / "config.json",
        checkpoint_path.parent / "config.json",
    ]
    config_path = None
    for c in config_candidates:
        if c.exists():
            config_path = c
            break
    if config_path is None:
        raise FileNotFoundError(f"config.json not found in {checkpoint_path}")

    # Find weights file
    weights_candidates = [
        checkpoint_path / "best_model.weights.h5",
        checkpoint_path / "best_model.h5",
    ]
    weights_path = None
    for w in weights_candidates:
        if w.exists():
            weights_path = w
            break
    if weights_path is None:
        raise FileNotFoundError(f"Weights file not found in {checkpoint_path}")

    return config_path, weights_path


def main():
    args = parse_args()

    # Import model components
    from model import (
        create_model,
        create_inference_model,
        CornerGAU,
        SimCCDecode,
        GlobalAveragePool2DAsAvgPool,
        AxisMean,
    )

    # Find checkpoint files
    checkpoint_path = Path(args.checkpoint)
    config_path, weights_path = find_checkpoint_files(checkpoint_path)

    print("=" * 60)
    print("Export TFLite FP16")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config:     {config_path}")
    print(f"Weights:    {weights_path}")

    # Load config
    with open(config_path) as f:
        cfg = json.load(f)

    print(f"\nModel config:")
    print(f"  backbone={cfg['backbone']} alpha={cfg['alpha']}")
    print(f"  img_size={cfg['img_size']} num_bins={cfg['num_bins']}")
    print(f"  fpn_ch={cfg['fpn_ch']} simcc_ch={cfg['simcc_ch']}")
    print(f"  use_gau={cfg.get('use_gau', False)}")
    if cfg.get('use_gau'):
        print(f"  gau_hidden_dim={cfg.get('gau_hidden_dim', 64)}")
        print(f"  fc_expansion_dim={cfg.get('fc_expansion_dim', 256)}")

    # Create model with GAU support
    train_model = create_model(
        backbone=cfg["backbone"],
        alpha=cfg["alpha"],
        fpn_ch=cfg["fpn_ch"],
        simcc_ch=cfg["simcc_ch"],
        img_size=cfg["img_size"],
        num_bins=cfg["num_bins"],
        tau=cfg.get("tau", 1.0),
        use_gau=cfg.get("use_gau", False),
        gau_hidden_dim=cfg.get("gau_hidden_dim", 64),
        fc_expansion_dim=cfg.get("fc_expansion_dim", 256),
    )

    # Load weights
    train_model.load_weights(str(weights_path))
    print(f"\nLoaded weights. Params: {train_model.count_params():,}")

    # Create inference model
    inference_model = create_inference_model(train_model)
    print(f"Inference model: {inference_model.input_shape} -> {inference_model.output_shape}")

    # Convert to TFLite FP16
    print("\nConverting to TFLite FP16...")
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / 1024 / 1024
    print(f"\nSaved: {output_path} ({size_mb:.2f} MB)")

    # Verify
    print("\nVerifying...")
    interpreter = tf.lite.Interpreter(model_path=str(output_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  Input:  {input_details[0]['shape']} {input_details[0]['dtype']}")
    for i, out in enumerate(output_details):
        print(f"  Output {i}: {out['shape']} {out['dtype']} name={out['name']}")

    # Test inference
    input_shape = input_details[0]["shape"]
    test_input = np.random.rand(*input_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()

    for i, out in enumerate(output_details):
        result = interpreter.get_tensor(out["index"])
        print(f"  Test output {i}: shape={result.shape}, range=[{result.min():.3f}, {result.max():.3f}]")

    print("\nDone!")


if __name__ == "__main__":
    main()
