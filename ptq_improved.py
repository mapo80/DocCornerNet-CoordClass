"""
Improved Post-Training Quantization using the float32 TFLite model.

This script:
1. Loads the float32 TFLite model
2. Calibrates with representative dataset (with augmentation)
3. Converts to int8 with better quantization parameters
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model", type=str, required=True,
                        help="Path to SavedModel directory")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset")
    parser.add_argument("--output", type=str, default="model_int8_ptq.tflite",
                        help="Output TFLite file")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of calibration samples")
    parser.add_argument("--img_size", type=int, default=224)
    return parser.parse_args()


def load_calibration_images(data_root, split='val', num_samples=1000, img_size=224):
    """Load calibration images with augmentation for better coverage."""
    data_root = Path(data_root)
    split_file = data_root / f"{split}.txt"

    with open(split_file) as f:
        filenames = [line.strip() for line in f if line.strip()]

    images_dir = data_root / "images"
    images = []

    print(f"Loading calibration images from {split}...")

    for fname in filenames:
        if len(images) >= num_samples * 4:  # Stop when we have enough with augmentation
            break

        img_path = images_dir / fname
        if not img_path.exists():
            continue

        try:
            img = tf.io.read_file(str(img_path))
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [img_size, img_size])
            img = tf.cast(img, tf.float32) / 255.0

            # ImageNet normalization
            mean = tf.constant([0.485, 0.456, 0.406])
            std = tf.constant([0.229, 0.224, 0.225])
            img = (img - mean) / std

            images.append(img.numpy())

            # Augmentations for better calibration coverage
            # Horizontal flip
            img_flip = tf.image.flip_left_right(img)
            images.append(img_flip.numpy())

            # Brightness variations
            for delta in [-0.15, 0.15]:
                img_bright = img + delta
                images.append(tf.clip_by_value(img_bright, -2.5, 2.5).numpy())

        except Exception as e:
            print(f"  Warning: Failed to load {fname}: {e}")
            continue

    print(f"Loaded {len(images)} calibration images (including augmentations)")
    return np.array(images, dtype=np.float32)


def main():
    args = parse_args()

    print("=" * 60)
    print("Improved Post-Training Quantization")
    print("=" * 60)
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU: {tf.config.list_physical_devices('GPU')}")

    # Load calibration data
    cal_images = load_calibration_images(
        args.data_root,
        split='val',
        num_samples=args.num_samples,
        img_size=args.img_size
    )

    # Representative dataset generator
    def representative_dataset():
        for i in range(len(cal_images)):
            yield [cal_images[i:i+1]]

    # Convert SavedModel to TFLite int8
    print(f"\nLoading SavedModel from {args.saved_model}...")
    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)

    # Full integer quantization settings
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    # Int8 with float fallback for unsupported ops
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    # Keep I/O as float32 for compatibility
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    print("\nConverting to int8...")
    try:
        tflite_model = converter.convert()

        output_path = Path(args.output)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"\nSaved: {output_path} ({size_mb:.2f} MB)")

        # Verify the model works
        print("\nVerifying model...")
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"Input: {input_details[0]['shape']} {input_details[0]['dtype']}")
        for od in output_details:
            print(f"Output: {od['name']} {od['shape']} {od['dtype']}")

        # Test inference
        test_input = cal_images[0:1]
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()

        print("\nTest inference successful!")

    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
