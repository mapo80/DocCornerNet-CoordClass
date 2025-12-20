"""
Export DocCornerNetV3 to various formats (TensorFlow/Keras).

Supports:
- SavedModel format
- TFLite (float32 and int8 quantized)
- ONNX (optional, requires tf2onnx)
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from model import create_model, create_inference_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export DocCornerNetV3 model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model weights (.h5) or SavedModel directory")
    parser.add_argument("--output_dir", type=str, default="./exported",
                        help="Output directory")
    parser.add_argument("--format", type=str, nargs="+",
                        default=["savedmodel", "tflite"],
                        choices=["savedmodel", "tflite", "tflite_int8", "onnx"],
                        help="Export formats")

    # Model config
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
    parser.add_argument("--num_bins", type=int, default=224,
                        help="Number of bins for coordinate classification")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Softmax temperature")

    # TFLite options
    parser.add_argument("--representative_data", type=str, default=None,
                        help="Path to representative data for int8 quantization")

    return parser.parse_args()


def _normalize_backbone_weights(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None
    return value


def _find_config_path(weights_path: Path) -> Optional[Path]:
    candidates = []
    if weights_path.is_dir():
        candidates.extend([weights_path / "config.json", weights_path.parent / "config.json"])
    else:
        candidates.extend([weights_path.parent / "config.json"])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_model_weights(args):
    """Load model with weights."""
    weights_path = Path(args.weights)

    # Try to load config
    config_path = _find_config_path(weights_path)
    if config_path and config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        backbone = config.get("backbone", args.backbone)
        backbone_minimalistic = config.get("backbone_minimalistic", args.backbone_minimalistic)
        backbone_include_preprocessing = config.get(
            "backbone_include_preprocessing", args.backbone_include_preprocessing
        )
        alpha = config.get("alpha", args.alpha)
        fpn_ch = config.get("fpn_ch", args.fpn_ch)
        simcc_ch = config.get("simcc_ch", args.simcc_ch)
        num_bins = config.get("num_bins", args.num_bins)
        img_size = config.get("img_size", args.img_size)
        tau = config.get("tau", args.tau)
        print(f"Loaded config from {config_path}")
    else:
        backbone = args.backbone
        backbone_minimalistic = args.backbone_minimalistic
        backbone_include_preprocessing = args.backbone_include_preprocessing
        alpha = args.alpha
        fpn_ch = args.fpn_ch
        simcc_ch = args.simcc_ch
        num_bins = args.num_bins
        img_size = args.img_size
        tau = args.tau

    def _try_load_serialized(p: Path):
        try:
            return keras.models.load_model(str(p), compile=False, safe_mode=False)
        except TypeError:
            return keras.models.load_model(str(p), compile=False)

    # If a full model was provided, load it directly.
    if weights_path.is_dir() or weights_path.suffix == ".keras":
        model = _try_load_serialized(weights_path)
        print(f"Loaded model from {weights_path}")
        try:
            inferred = model.input_shape[1]
            if isinstance(inferred, int):
                img_size = inferred
        except Exception:
            pass
        return model, img_size

    # Otherwise, build training model, load weights, then derive inference model.
    train_model = create_model(
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

    if weights_path.suffix == ".h5":
        train_model.load_weights(str(weights_path))
        print(f"Loaded weights from {weights_path}")
    else:
        raise ValueError(f"Unsupported weights file: {weights_path}")

    model = create_inference_model(train_model)

    return model, img_size


def export_savedmodel(model, output_path: Path, img_size: int):
    """Export to SavedModel format."""
    print(f"\nExporting SavedModel to {output_path}...")

    if hasattr(model, "export"):
        model.export(str(output_path))
    else:
        model.save(str(output_path), save_format="tf")

    # Get model size
    size_mb = sum(
        f.stat().st_size for f in output_path.rglob("*") if f.is_file()
    ) / (1024 * 1024)

    print(f"  SavedModel size: {size_mb:.2f} MB")
    return size_mb


def export_tflite(model, output_path: Path, img_size: int, quantize: bool = False,
                  representative_data_path: str = None):
    """Export to TFLite format."""
    print(f"\nExporting TFLite ({'int8' if quantize else 'float32'}) to {output_path}...")

    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Representative dataset for full int8
        if representative_data_path:
            def representative_dataset():
                # Load some sample images
                data_path = Path(representative_data_path)
                images = list(data_path.glob("*.jpg"))[:100]

                for img_path in images:
                    img = tf.io.read_file(str(img_path))
                    img = tf.image.decode_jpeg(img, channels=3)
                    img = tf.image.resize(img, [img_size, img_size])
                    img = tf.cast(img, tf.float32) / 255.0
                    # ImageNet normalization
                    mean = [0.485, 0.456, 0.406]
                    std = [0.229, 0.224, 0.225]
                    img = (img - mean) / std
                    img = tf.expand_dims(img, 0)
                    yield [img]

            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.float32
        else:
            # Dynamic range quantization
            print("  Warning: No representative data, using dynamic range quantization")

    # Convert
    tflite_model = converter.convert()

    # Save
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  TFLite size: {size_mb:.2f} MB")

    return size_mb


def export_onnx(model, output_path: Path, img_size: int):
    """Export to ONNX format (requires tf2onnx)."""
    try:
        import tf2onnx
    except ImportError:
        print("  Warning: tf2onnx not installed, skipping ONNX export")
        return None

    print(f"\nExporting ONNX to {output_path}...")

    # Convert
    spec = (tf.TensorSpec((1, img_size, img_size, 3), tf.float32, name="input"),)

    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        output_path=str(output_path),
        opset=13,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ONNX size: {size_mb:.2f} MB")

    return size_mb


def benchmark_tflite(tflite_path: Path, img_size: int, num_runs: int = 100):
    """Benchmark TFLite inference speed."""
    print(f"\nBenchmarking TFLite ({tflite_path.name})...")

    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Determine input type
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.int8:
        # Quantized model
        input_scale = input_details[0]['quantization'][0]
        input_zero_point = input_details[0]['quantization'][1]
        dummy_input = np.random.randint(-128, 127, (1, img_size, img_size, 3)).astype(np.int8)
    else:
        dummy_input = np.random.randn(1, img_size, img_size, 3).astype(np.float32)

    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        times.append(time.perf_counter() - start)

    times_ms = np.array(times) * 1000

    print(f"  Mean: {np.mean(times_ms):.2f} ms")
    print(f"  Std:  {np.std(times_ms):.2f} ms")
    print(f"  P50:  {np.percentile(times_ms, 50):.2f} ms")
    print(f"  P95:  {np.percentile(times_ms, 95):.2f} ms")

    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95)),
    }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DocCornerNetV3 Export")
    print("=" * 60)

    # Load model
    model, img_size = load_model_weights(args)
    print(f"Model parameters: {model.count_params():,}")

    # Test forward pass
    dummy_input = np.random.randn(1, img_size, img_size, 3).astype(np.float32)
    outputs = model(dummy_input, training=False)
    print(f"\nTest forward pass:")
    print(f"  coords shape: {outputs['coords'].shape}")
    print(f"  score shape: {outputs['score'].shape}")

    results = {"formats": {}}

    # Export SavedModel
    if "savedmodel" in args.format:
        savedmodel_path = output_dir / "savedmodel"
        size = export_savedmodel(model, savedmodel_path, img_size)
        results["formats"]["savedmodel"] = {"size_mb": size}

    # Export TFLite (float32)
    if "tflite" in args.format:
        tflite_path = output_dir / "model_float32.tflite"
        size = export_tflite(model, tflite_path, img_size, quantize=False)
        results["formats"]["tflite_float32"] = {"size_mb": size}

        # Benchmark
        bench = benchmark_tflite(tflite_path, img_size)
        results["formats"]["tflite_float32"]["benchmark"] = bench

    # Export TFLite (int8)
    if "tflite_int8" in args.format:
        tflite_int8_path = output_dir / "model_int8.tflite"
        size = export_tflite(
            model, tflite_int8_path, img_size,
            quantize=True,
            representative_data_path=args.representative_data
        )
        results["formats"]["tflite_int8"] = {"size_mb": size}

        # Benchmark
        bench = benchmark_tflite(tflite_int8_path, img_size)
        results["formats"]["tflite_int8"]["benchmark"] = bench

    # Export ONNX
    if "onnx" in args.format:
        onnx_path = output_dir / "model.onnx"
        size = export_onnx(model, onnx_path, img_size)
        if size:
            results["formats"]["onnx"] = {"size_mb": size}

    # Save results
    with open(output_dir / "export_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Export Complete")
    print("=" * 60)
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
