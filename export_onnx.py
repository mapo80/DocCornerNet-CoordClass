#!/usr/bin/env python3
"""
Export DocCornerNetV3 to ONNX format.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from model import create_model, create_inference_model


def parse_args():
    parser = argparse.ArgumentParser(description="Export DocCornerNetV3 to ONNX")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                        help="Path to checkpoint directory or weights file")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=13,
                        help="ONNX opset version")
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)

    # Find config and weights
    if checkpoint_path.is_dir():
        config_path = checkpoint_path / "config.json"
        weights_path = checkpoint_path / "best_model.weights.h5"
    else:
        config_path = checkpoint_path.parent / "config.json"
        weights_path = checkpoint_path

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    print(f"Loaded config from {config_path}")
    print(f"  backbone: {config.get('backbone')}")
    print(f"  alpha: {config.get('alpha')}")
    print(f"  img_size: {config.get('img_size')}")

    # Create model
    train_model = create_model(
        backbone=config.get("backbone", "mobilenetv2"),
        alpha=config.get("alpha", 0.35),
        backbone_minimalistic=config.get("backbone_minimalistic", False),
        backbone_include_preprocessing=config.get("backbone_include_preprocessing", False),
        backbone_weights=None,
        fpn_ch=config.get("fpn_ch", 32),
        simcc_ch=config.get("simcc_ch", 96),
        img_size=config.get("img_size", 256),
        num_bins=config.get("num_bins", 256),
        tau=config.get("tau", 1.0),
    )

    # Load weights
    train_model.load_weights(str(weights_path))
    print(f"Loaded weights from {weights_path}")
    print(f"Model parameters: {train_model.count_params():,}")

    # Create inference model
    model = create_inference_model(train_model)

    # Test forward pass
    img_size = config.get("img_size", 256)
    dummy_input = np.random.randn(1, img_size, img_size, 3).astype(np.float32)
    outputs = model(dummy_input, training=False)
    print(f"\nTest forward pass:")
    print(f"  coords shape: {outputs[0].shape}")
    print(f"  score shape: {outputs[1].shape}")

    # Export to ONNX
    try:
        import tf2onnx
    except ImportError:
        print("Error: tf2onnx not installed. Run: pip install tf2onnx")
        return

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = checkpoint_path / "model.onnx" if checkpoint_path.is_dir() else checkpoint_path.parent / "model.onnx"

    print(f"\nExporting ONNX to {output_path}...")

    spec = (tf.TensorSpec((1, img_size, img_size, 3), tf.float32, name="input"),)

    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        output_path=str(output_path),
        opset=args.opset,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"ONNX export complete!")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Path: {output_path}")


if __name__ == "__main__":
    main()
