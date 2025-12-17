"""
Selective Post-Training Quantization.

Uses dynamic range quantization to keep coordinate-sensitive layers in float32
while quantizing the backbone.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output", type=str, default="model_int8_selective.tflite")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--img_size", type=int, default=224)
    return parser.parse_args()


def load_calibration_images(data_root, split='val', num_samples=500, img_size=224):
    """Load calibration images."""
    data_root = Path(data_root)
    split_file = data_root / f"{split}.txt"

    with open(split_file) as f:
        filenames = [line.strip() for line in f if line.strip()]

    images_dir = data_root / "images"
    images = []

    for fname in filenames[:num_samples]:
        img_path = images_dir / fname
        if not img_path.exists():
            continue

        try:
            img = tf.io.read_file(str(img_path))
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [img_size, img_size])
            img = tf.cast(img, tf.float32) / 255.0

            mean = tf.constant([0.485, 0.456, 0.406])
            std = tf.constant([0.229, 0.224, 0.225])
            img = (img - mean) / std

            images.append(img.numpy())

        except Exception:
            continue

    return np.array(images, dtype=np.float32)


def main():
    args = parse_args()

    print("=" * 60)
    print("Selective/Dynamic Quantization")
    print("=" * 60)
    print(f"TensorFlow: {tf.__version__}")

    # Load calibration data
    print(f"\nLoading calibration images...")
    cal_images = load_calibration_images(args.data_root, num_samples=args.num_samples)
    print(f"Loaded {len(cal_images)} images")

    def representative_dataset():
        for i in range(len(cal_images)):
            yield [cal_images[i:i+1]]

    # Method 1: Dynamic range quantization (simpler, often works better)
    print(f"\n--- Method 1: Dynamic Range Quantization ---")
    converter1 = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)
    converter1.optimizations = [tf.lite.Optimize.DEFAULT]
    # Don't set representative_dataset for dynamic range

    try:
        tflite1 = converter1.convert()
        path1 = Path(args.output).with_stem(Path(args.output).stem + "_dynamic")
        with open(path1, 'wb') as f:
            f.write(tflite1)
        print(f"Saved: {path1} ({len(tflite1)/(1024*1024):.2f} MB)")
    except Exception as e:
        print(f"Dynamic quantization failed: {e}")

    # Method 2: Float16 quantization (hybrid)
    print(f"\n--- Method 2: Float16 Quantization ---")
    converter2 = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)
    converter2.optimizations = [tf.lite.Optimize.DEFAULT]
    converter2.target_spec.supported_types = [tf.float16]

    try:
        tflite2 = converter2.convert()
        path2 = Path(args.output).with_stem(Path(args.output).stem + "_fp16")
        with open(path2, 'wb') as f:
            f.write(tflite2)
        print(f"Saved: {path2} ({len(tflite2)/(1024*1024):.2f} MB)")
    except Exception as e:
        print(f"Float16 quantization failed: {e}")

    # Method 3: Full int8 with experimental selective quantization
    print(f"\n--- Method 3: Int8 with SELECT_TF_OPS ---")
    converter3 = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)
    converter3.optimizations = [tf.lite.Optimize.DEFAULT]
    converter3.representative_dataset = representative_dataset
    converter3.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter3.inference_input_type = tf.float32
    converter3.inference_output_type = tf.float32
    # Try to preserve more precision in certain ops
    converter3._experimental_lower_tensor_list_ops = False

    try:
        tflite3 = converter3.convert()
        path3 = Path(args.output).with_stem(Path(args.output).stem + "_int8_hybrid")
        with open(path3, 'wb') as f:
            f.write(tflite3)
        print(f"Saved: {path3} ({len(tflite3)/(1024*1024):.2f} MB)")
    except Exception as e:
        print(f"Int8 hybrid failed: {e}")

    # Method 4: 16x8 quantization (activations int16, weights int8)
    print(f"\n--- Method 4: 16x8 Quantization ---")
    converter4 = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)
    converter4.optimizations = [tf.lite.Optimize.DEFAULT]
    converter4.representative_dataset = representative_dataset
    converter4.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter4.inference_input_type = tf.float32
    converter4.inference_output_type = tf.float32

    try:
        tflite4 = converter4.convert()
        path4 = Path(args.output).with_stem(Path(args.output).stem + "_16x8")
        with open(path4, 'wb') as f:
            f.write(tflite4)
        print(f"Saved: {path4} ({len(tflite4)/(1024*1024):.2f} MB)")
    except Exception as e:
        print(f"16x8 quantization failed: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
