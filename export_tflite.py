"""
Export DocCornerNet models to TFLite format.

Usage:
    # Export teacher model
    python export_tflite.py \
        --model_path checkpoints/best_model.weights.h5 \
        --output exported_tflite/doccornernet_v3_224_float32.tflite

    # Export with float16 quantization
    python export_tflite.py \
        --model_path checkpoints/best_model.weights.h5 \
        --output exported_tflite/doccornernet_v3_224_float16.tflite \
        --float16

    # Export student model
    python export_tflite.py \
        --model_path checkpoints_student/student_distill_*/best_student.weights.h5 \
        --config checkpoints_student/student_distill_*/config.json \
        --output exported_tflite/doccornernet_v3_student_224_float16.tflite \
        --float16
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from model import create_model

# ImageNet normalization constants (RGB, [0,1])
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export DocCornerNet to TFLite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model weights (.h5) or SavedModel directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.json (auto-detected if not specified)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output TFLite file path",
    )
    parser.add_argument(
        "--float16",
        action="store_true",
        help="Apply float16 quantization",
    )
    parser.add_argument(
        "--tflite_input_norm",
        type=str,
        default="auto",
        choices=["auto", "imagenet", "zero_one", "raw255"],
        help=(
            "Expected input normalization for the exported TFLite model. "
            "'auto' means: do not add extra preprocessing (input must match training). "
            "Use 'raw255' to export a model that accepts raw float32 pixels in [0,255]."
        ),
    )

    # Model architecture (used if config not found)
    parser.add_argument("--backbone", type=str, default="mobilenetv3_small")
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--backbone_minimalistic", action="store_true")
    parser.add_argument(
        "--backbone_include_preprocessing",
        action="store_true",
        help="Enable built-in backbone preprocessing (legacy checkpoints may require this)",
    )
    parser.add_argument("--fpn_ch", type=int, default=48)
    parser.add_argument("--simcc_ch", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_bins", type=int, default=224)
    parser.add_argument("--tau", type=float, default=1.0)

    return parser.parse_args()


def find_config(model_path: Path, explicit_config: str = None) -> dict:
    """Find and load config.json."""
    if explicit_config:
        config_path = Path(explicit_config)
    else:
        # Try common locations
        candidates = [
            model_path.parent / "config.json",
            model_path.parent.parent / "config.json",
        ]
        config_path = None
        for c in candidates:
            if c.exists():
                config_path = c
                break

    if config_path and config_path.exists():
        with open(config_path) as f:
            raw_config = json.load(f)
        print(f"Loaded config from {config_path}")

        # Check if this is a student distillation config (has student_* keys)
        is_student = "student_alpha" in raw_config or "student_fpn_ch" in raw_config

        if is_student:
            # Map student parameters to standard names
            config = {
                "backbone": raw_config.get("student_backbone", raw_config.get("backbone", "mobilenetv3_small")),
                "alpha": raw_config.get("student_alpha", 0.75),
                "backbone_minimalistic": raw_config.get("student_backbone_minimalistic", False),
                "backbone_include_preprocessing": raw_config.get(
                    "student_backbone_include_preprocessing",
                    raw_config.get("backbone_include_preprocessing", False),
                ),
                "fpn_ch": raw_config.get("student_fpn_ch", 32),
                "simcc_ch": raw_config.get("student_simcc_ch", 96),
                "img_size": raw_config.get("img_size", 224),
                "num_bins": raw_config.get("num_bins", 224),
                "tau": raw_config.get("tau", 1.0),
            }
            print(f"  Detected student config: alpha={config['alpha']}, fpn_ch={config['fpn_ch']}, simcc_ch={config['simcc_ch']}")
        else:
            config = raw_config

        return config

    return {}


def create_inference_model(model, single_output: bool = True):
    """
    Create inference model for TFLite export.

    Args:
        model: The training model
        single_output: If True, concatenate coords and score into single [1, 9] output.
                      If False, output [coords, score] as separate tensors.
    """
    raise RuntimeError("create_inference_model is deprecated; use create_tflite_inference_model().")


def _apply_norm_transform(x, src_norm: str, dst_norm: str):
    """
    Transform input tensor x from src_norm to dst_norm.

    Supported norms:
      - raw255: float32 in [0,255]
      - zero_one: float32 in [0,1]
      - imagenet: float32 standardized with ImageNet mean/std (as in dataset.py)
    """
    src = (src_norm or "imagenet").lower().strip()
    dst = (dst_norm or "imagenet").lower().strip()

    if src == dst:
        return x

    def to_zero_one(t, src_mode: str):
        if src_mode in {"raw255", "0_255", "0255"}:
            return keras.layers.Rescaling(1.0 / 255.0, name="in_rescale_255_to_01")(t)
        if src_mode in {"zero_one", "0_1", "01"}:
            return t
        if src_mode == "imagenet":
            # x = (z * std) + mean  (still in [0,1] domain)
            std = tf.constant(IMAGENET_STD.reshape(1, 1, 1, 3), dtype=tf.float32)
            mean = tf.constant(IMAGENET_MEAN.reshape(1, 1, 1, 3), dtype=tf.float32)
            t = keras.layers.Multiply(name="in_imagenet_to_01_mul_std")([t, std])
            return keras.layers.Add(name="in_imagenet_to_01_add_mean")([t, mean])
        raise ValueError(f"Unsupported src_norm='{src_mode}'")

    if dst in {"zero_one", "0_1", "01"}:
        return to_zero_one(x, src)

    if dst in {"raw255", "0_255", "0255"}:
        z = to_zero_one(x, src)
        return keras.layers.Rescaling(255.0, name="in_rescale_01_to_255")(z)

    if dst == "imagenet":
        z = to_zero_one(x, src)
        # Normalize (z - mean) / std
        var = (IMAGENET_STD ** 2).astype(np.float32)
        return keras.layers.Normalization(
            axis=-1,
            mean=IMAGENET_MEAN,
            variance=var,
            name="in_norm_imagenet",
        )(z)

    raise ValueError(f"Unsupported dst_norm='{dst_norm}'")


def create_tflite_inference_model(
    model: keras.Model,
    img_size: int,
    model_input_norm: str,
    tflite_input_norm: str,
    single_output: bool = True,
):
    """
    Create an inference model suitable for TFLite export.

    - Adds an optional preprocessing stack to map from `tflite_input_norm` -> `model_input_norm`.
    - Applies sigmoid to the score output.
    - Optionally concatenates outputs to a single [B, 9] tensor.
    """
    inputs = keras.Input(shape=(img_size, img_size, 3), dtype=tf.float32, name="image")

    x = _apply_norm_transform(inputs, tflite_input_norm, model_input_norm)
    outputs = model(x, training=False)

    if isinstance(outputs, dict):
        coords = outputs["coords"]
        score_logit = outputs["score_logit"]
    else:
        coords = outputs[0]
        score_logit = outputs[1]

    score = keras.ops.sigmoid(score_logit)

    if single_output:
        output = keras.ops.concatenate([coords, score], axis=-1)
        return keras.Model(inputs=inputs, outputs=output, name="doccornernet_inference")

    return keras.Model(inputs=inputs, outputs=[coords, score], name="doccornernet_inference")

    # Get model outputs
    outputs = model(inputs, training=False)

    # Extract coords and score_logit
    if isinstance(outputs, dict):
        coords = outputs["coords"]
        score_logit = outputs["score_logit"]
    else:
        coords = outputs[0]
        score_logit = outputs[1]

    # Apply sigmoid to score using keras.ops (required for Keras 3)
    score = keras.ops.sigmoid(score_logit)

    if single_output:
        # Concatenate to single [B, 9] output: [x0, y0, x1, y1, x2, y2, x3, y3, score]
        # This is compatible with web/mobile runtimes expecting [1, 9]
        output = keras.ops.concatenate([coords, score], axis=-1)
        inference_model = keras.Model(
            inputs=inputs,
            outputs=output,
            name="doccornernet_inference",
        )
    else:
        # Separate outputs for coords [B, 8] and score [B, 1]
        inference_model = keras.Model(
            inputs=inputs,
            outputs=[coords, score],
            name="doccornernet_inference",
        )

    return inference_model


def export_tflite(
    model,
    output_path: str,
    float16: bool = False,
    img_size: int = 224,
    model_input_norm: str = "imagenet",
    tflite_input_norm: str = "auto",
):
    """Export model to TFLite format."""
    model_input_norm = (model_input_norm or "imagenet").lower().strip()
    if tflite_input_norm == "auto":
        tflite_input_norm = model_input_norm
    tflite_input_norm = (tflite_input_norm or model_input_norm).lower().strip()

    # Create inference model with single [1, 9] output
    inference_model = create_tflite_inference_model(
        model=model,
        img_size=img_size,
        model_input_norm=model_input_norm,
        tflite_input_norm=tflite_input_norm,
        single_output=True,
    )

    # Get concrete function
    @tf.function(input_signature=[tf.TensorSpec([1, img_size, img_size, 3], tf.float32)])
    def inference_fn(x):
        # Single output: [x0, y0, x1, y1, x2, y2, x3, y3, score]
        return inference_model(x, training=False)

    concrete_func = inference_fn.get_concrete_function()

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    if float16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("Applying float16 quantization...")

    # Convert
    tflite_model = converter.convert()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"Saved TFLite model to {output_path} ({size_mb:.2f} MB)")

    return output_path


def verify_tflite(tflite_path: str, img_size: int):
    """Verify the exported TFLite model."""
    print(f"\nVerifying {tflite_path}...")

    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  Input: {input_details[0]['shape']} {input_details[0]['dtype']}")
    for i, out in enumerate(output_details):
        print(f"  Output {i} ({out['name']}): {out['shape']} {out['dtype']}")

    # Validate expected output shape [1, 9]
    if len(output_details) == 1 and list(output_details[0]['shape']) == [1, 9]:
        print(f"  ✓ Output shape [1, 9] matches expected format")
    else:
        print(f"  ⚠ Warning: Expected single output with shape [1, 9]")

    # Test inference
    test_input = np.random.rand(1, img_size, img_size, 3).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()

    # Get output
    output = interpreter.get_tensor(output_details[0]["index"])
    print(f"  Test inference successful!")
    print(f"    Output shape: {output.shape}")
    print(f"    Coords [0:8]: range=[{output[0, :8].min():.3f}, {output[0, :8].max():.3f}]")
    print(f"    Score [8]:    {output[0, 8]:.3f}")

    # Benchmark
    import time

    n_runs = 100
    start = time.time()
    for _ in range(n_runs):
        interpreter.set_tensor(input_details[0]["index"], test_input)
        interpreter.invoke()
    elapsed = time.time() - start
    avg_ms = (elapsed / n_runs) * 1000
    print(f"  Inference time: {avg_ms:.2f} ms (avg over {n_runs} runs)")


def main():
    args = parse_args()

    model_path = Path(args.model_path)
    print(f"Loading model from {model_path}...")

    # Load config
    config = find_config(model_path, args.config)

    # Get model parameters
    backbone = config.get("backbone", args.backbone)
    alpha = config.get("alpha", args.alpha)
    backbone_minimalistic = config.get("backbone_minimalistic", args.backbone_minimalistic)
    backbone_include_preprocessing = args.backbone_include_preprocessing or config.get(
        "backbone_include_preprocessing", False
    )
    fpn_ch = config.get("fpn_ch", args.fpn_ch)
    simcc_ch = config.get("simcc_ch", args.simcc_ch)
    img_size = config.get("img_size", args.img_size)
    num_bins = config.get("num_bins", args.num_bins)
    tau = config.get("tau", args.tau)
    model_input_norm = config.get("input_norm", "imagenet")

    print(f"Model config:")
    print(f"  backbone: {backbone}")
    print(f"  alpha: {alpha}")
    print(f"  backbone_minimalistic: {backbone_minimalistic}")
    print(f"  backbone_include_preprocessing: {backbone_include_preprocessing}")
    print(f"  fpn_ch: {fpn_ch}")
    print(f"  simcc_ch: {simcc_ch}")
    print(f"  img_size: {img_size}")
    print(f"  num_bins: {num_bins}")
    print(f"  tau: {tau}")
    print(f"  model_input_norm: {model_input_norm}")
    if args.tflite_input_norm == "auto":
        print(f"  tflite_input_norm: auto (no extra preprocessing)")
    else:
        print(f"  tflite_input_norm: {args.tflite_input_norm}")

    # Create model
    model = create_model(
        backbone=backbone,
        alpha=alpha,
        backbone_minimalistic=backbone_minimalistic,
        backbone_include_preprocessing=backbone_include_preprocessing,
        backbone_weights=None,  # Don't download weights
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
    elif model_path.is_dir():
        # Try to find weights in directory
        for weights_file in ["best_model.weights.h5", "best_student.weights.h5", "final_model.weights.h5"]:
            weights_path = model_path / weights_file
            if weights_path.exists():
                model.load_weights(str(weights_path))
                print(f"Loaded weights from {weights_path}")
                break
        else:
            raise ValueError(f"No weights found in {model_path}")

    print(f"Model parameters: {model.count_params():,}")

    # Export
    print(f"\nExporting to TFLite...")
    output_path = export_tflite(
        model,
        args.output,
        float16=args.float16,
        img_size=img_size,
        model_input_norm=model_input_norm,
        tflite_input_norm=args.tflite_input_norm,
    )

    # Verify
    verify_tflite(output_path, img_size)

    print("\nDone!")


if __name__ == "__main__":
    main()
