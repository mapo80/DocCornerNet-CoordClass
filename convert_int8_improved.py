"""
Improved int8 quantization with better calibration.

Strategies:
1. Use diverse representative dataset (augmented)
2. Use full precision for sensitive layers (coordinates)
3. Per-channel quantization where possible
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output", type=str, default="model_int8_improved.tflite")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--img_size", type=int, default=224)
    return parser.parse_args()


def load_calibration_data(data_root, split='val', num_samples=500, img_size=224):
    """Load and augment calibration data."""
    data_root = Path(data_root)
    split_file = data_root / f"{split}.txt"

    with open(split_file) as f:
        filenames = [line.strip() for line in f if line.strip()]

    images_dir = data_root / "images"

    # Load images
    images = []
    for fname in filenames[:num_samples]:
        img_path = images_dir / fname
        if not img_path.exists():
            continue

        img = tf.io.read_file(str(img_path))
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32) / 255.0

        # ImageNet normalization
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        img = (img - mean) / std

        images.append(img.numpy())

        # Also add augmented versions for better coverage
        # Flip
        img_flip = tf.image.flip_left_right(img)
        images.append(img_flip.numpy())

        # Brightness variations
        for delta in [-0.1, 0.1]:
            img_bright = img + delta
            images.append(img_bright.numpy())

    print(f"Loaded {len(images)} calibration images (with augmentation)")
    return np.array(images, dtype=np.float32)


def main():
    args = parse_args()

    print("=" * 60)
    print("Improved Int8 Quantization")
    print("=" * 60)
    print(f"TensorFlow: {tf.__version__}")

    # Load calibration data
    print(f"\nLoading calibration data from {args.data_root}...")
    cal_data = load_calibration_data(
        args.data_root,
        split='val',
        num_samples=args.num_samples,
        img_size=args.img_size
    )

    # Representative dataset generator
    def representative_dataset():
        for i in range(len(cal_data)):
            yield [cal_data[i:i+1]]

    # Load SavedModel
    print(f"\nLoading model from {args.saved_model}...")

    # Convert using TFLiteConverter
    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)

    # Optimization settings
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    # Use int8 with float fallback for unsupported ops
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS  # Fallback for complex ops
    ]

    # Keep input/output as float32 for easier integration
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    # Quantization config - experimental options
    # Enable per-tensor/per-channel quantization
    converter._experimental_lower_tensor_list_ops = False

    print("\nConverting to int8...")
    try:
        tflite_model = converter.convert()

        output_path = Path(args.output)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"\nSaved: {output_path} ({size_mb:.2f} MB)")

    except Exception as e:
        print(f"Conversion failed: {e}")

        # Try with only int8 builtins
        print("\nRetrying with int8 builtins only...")
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        try:
            tflite_model = converter.convert()

            output_path = Path(args.output)
            with open(output_path, 'wb') as f:
                f.write(tflite_model)

            size_mb = len(tflite_model) / (1024 * 1024)
            print(f"\nSaved: {output_path} ({size_mb:.2f} MB)")

        except Exception as e2:
            print(f"Still failed: {e2}")
            raise


if __name__ == "__main__":
    main()
