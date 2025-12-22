"""
Export a DocCornerNet checkpoint to a quantized TFLite model (PTQ).

This script is tailored for `checkpoints/mobilenetv2_256_best` but works for any
checkpoint folder/weights that include a compatible `config.json`.

Examples:
  # Int8 PTQ with float32 I/O (easy integration; internal int8 subgraphs)
  python export_tflite_int8.py \
    --checkpoint checkpoints/mobilenetv2_256_best \
    --data_root /path/to/doc-scanner-dataset-labeled \
    --split val_cleaned \
    --output exported_tflite/doccornernet_v3_mnv2_256_best_int8.tflite

  # Same, but allow float fallback for ops that don't quantize well (no SELECT_TF_OPS)
  python export_tflite_int8.py \
    --checkpoint checkpoints/mobilenetv2_256_best \
    --data_root /path/to/doc-scanner-dataset-labeled \
    --split val_cleaned \
    --output exported_tflite/doccornernet_v3_mnv2_256_best_int8_hybrid.tflite \
    --allow_float_fallback
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from dataset import create_dataset
from model import create_model

# ImageNet normalization constants (RGB, [0,1])
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(
        description="Export DocCornerNet to quantized TFLite (PTQ)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint directory (containing config.json + best_model.weights.h5) or weights .h5 file",
    )
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Dataset root for representative dataset calibration",
    )
    p.add_argument(
        "--split",
        type=str,
        default="val_cleaned",
        help="Split used for representative dataset (e.g. val_cleaned, train_cleaned_plus_outliers)",
    )
    p.add_argument("--num_calib", type=int, default=500, help="Number of calibration samples")
    p.add_argument(
        "--tflite_input_norm",
        type=str,
        default="auto",
        choices=["auto", "imagenet", "zero_one", "raw255"],
        help="Input normalization expected by the exported TFLite model",
    )
    p.add_argument(
        "--quantization",
        type=str,
        default="int8",
        choices=["int8", "int16x8", "dynamic"],
        help=(
            "Quantization scheme. "
            "'int8' = full-int8 where possible, "
            "'int16x8' = activations int16 + weights int8 (often higher accuracy), "
            "'dynamic' = dynamic range (weights-only) quantization."
        ),
    )
    p.add_argument(
        "--io_dtype",
        type=str,
        default="float32",
        choices=["float32", "uint8", "int8"],
        help="TFLite model input dtype",
    )
    p.add_argument(
        "--output_dtype",
        type=str,
        default="float32",
        choices=["float32", "uint8", "int8"],
        help="TFLite model output dtype",
    )
    p.add_argument(
        "--output_format",
        type=str,
        default="coords9",
        choices=["coords9", "simcc_logits"],
        help=(
            "TFLite output format. "
            "'coords9' outputs a single [B,9] tensor: [x0..y3, score]. "
            "'simcc_logits' outputs [simcc_x, simcc_y, score_logit] and expects decoding outside the model."
        ),
    )
    p.add_argument(
        "--simcc_packed_layout",
        type=str,
        default="8_first",
        choices=["8_first", "bins_first"],
        help=(
            "Only for output_format=simcc_logits: "
            "'8_first' packs logits as [B,8,num_bins] (X corners then Y corners), "
            "'bins_first' packs logits as [B,num_bins,8] to avoid TRANSPOSE ops."
        ),
    )
    p.add_argument(
        "--allow_float_fallback",
        action="store_true",
        help="Allow float TFLite builtins fallback for ops that can't be quantized (avoids SELECT_TF_OPS)",
    )
    p.add_argument(
        "--axis_mean_impl",
        type=str,
        default="dwconv_full",
        choices=["mean", "avgpool", "dwconv_full", "dwconv_strided", "dwconv_pyramid"],
        help="Axis reduction implementation used for x/y marginals in the export model (XNNPACK coverage vs speed).",
    )
    p.add_argument(
        "--global_pool_impl",
        type=str,
        default="dwconv_full",
        choices=["mean", "avgpool", "dwconv_full", "dwconv_strided", "dwconv_pyramid"],
        help="Global pooling implementation used for simcc_global_gap/score_gap in the export model (XNNPACK coverage vs speed).",
    )
    p.add_argument(
        "--score_pool_impl",
        type=str,
        default="auto",
        choices=["auto", "mean", "avgpool", "dwconv_full", "dwconv_strided", "dwconv_pyramid"],
        help="Override pooling impl for score_gap only (default: use --global_pool_impl).",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output TFLite file path",
    )
    p.add_argument("--threads", type=int, default=1, help="Threads for verification interpreter")
    return p.parse_args()


def find_config(checkpoint_path: Path) -> dict:
    if checkpoint_path.is_dir():
        candidates = [checkpoint_path / "config.json", checkpoint_path.parent / "config.json"]
    else:
        candidates = [checkpoint_path.parent / "config.json"]

    for c in candidates:
        if c.exists():
            with open(c) as f:
                cfg = json.load(f)
            print(f"Loaded config from {c}")
            return cfg

    return {}


def find_weights(checkpoint_path: Path) -> Path:
    if checkpoint_path.suffix == ".h5":
        return checkpoint_path
    if checkpoint_path.is_dir():
        for name in ("best_model.weights.h5", "final_model.weights.h5", "best_student.weights.h5"):
            p = checkpoint_path / name
            if p.exists():
                return p
    raise FileNotFoundError(f"Could not find weights under {checkpoint_path}")


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
        var = (IMAGENET_STD**2).astype(np.float32)
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
    output_format: str = "coords9",
    simcc_packed_layout: str = "8_first",
) -> keras.Model:
    """
    Build a TFLite-friendly inference model:
    - Optional input preprocessing: `tflite_input_norm` -> `model_input_norm`
    - Output: single [B, 9] tensor: [x0..y3, score]
    """
    inputs = keras.Input(shape=(img_size, img_size, 3), dtype=tf.float32, name="image")
    x = _apply_norm_transform(inputs, tflite_input_norm, model_input_norm)

    outputs = model(x, training=False)

    if isinstance(outputs, dict):
        simcc_x = outputs["simcc_x"]
        simcc_y = outputs["simcc_y"]
        score_logit = outputs["score_logit"]
        coords = outputs["coords"]
    else:
        # Legacy: [coords, score_logit]
        simcc_x = None
        simcc_y = None
        coords = outputs[0]
        score_logit = outputs[1]

    fmt = (output_format or "coords9").lower().strip()
    if fmt == "coords9":
        score = keras.ops.sigmoid(score_logit)
        out = keras.ops.concatenate([coords, score], axis=-1)
        return keras.Model(inputs=inputs, outputs=out, name="doccornernet_inference")

    if fmt == "simcc_logits":
        if simcc_x is None or simcc_y is None:
            raise ValueError("output_format='simcc_logits' requires model dict outputs with simcc_x/simcc_y.")
        # NOTE: we decode outside the model; keep logits as-is (can be quantized).
        layout = str(simcc_packed_layout).lower().strip()
        if layout == "bins_first":
            # simcc_x/y: [B, num_bins, 4] -> simcc_xy: [B, num_bins, 8]
            simcc_xy = keras.ops.concatenate([simcc_x, simcc_y], axis=-1)
        elif layout == "8_first":
            # simcc_x/y: [B, 4, num_bins] -> simcc_xy: [B, 8, num_bins]
            simcc_xy = keras.ops.concatenate([simcc_x, simcc_y], axis=1)
        else:
            raise ValueError(f"Unsupported simcc_packed_layout='{simcc_packed_layout}'")
        return keras.Model(
            inputs=inputs,
            outputs=[simcc_xy, score_logit],
            name="doccornernet_simcc_logits",
        )

    raise ValueError(f"Unsupported output_format='{output_format}'")


def _io_dtype(value: str):
    v = (value or "float32").lower().strip()
    if v == "float32":
        return tf.float32
    if v == "uint8":
        return tf.uint8
    if v == "int8":
        return tf.int8
    raise ValueError(f"Unsupported io_dtype='{value}'")


def _copy_weights_by_path(src: keras.Model, dst: keras.Model) -> None:
    """
    Copy weights from src to dst by variable path.

    This is more robust than `set_weights()` when model graph traversal order changes
    (e.g., when toggling non-weight layers like Permute/Reshape).
    """
    src_map = {w.path: w.numpy() for w in src.weights}
    missing = []
    mismatched = []

    for w in dst.weights:
        v = src_map.get(w.path)
        if v is None:
            missing.append(w.path)
            continue
        if tuple(w.shape) != tuple(v.shape):
            mismatched.append((w.path, tuple(w.shape), tuple(v.shape)))
            continue
        w.assign(v)

    if mismatched:
        msg = "\n".join(f"  {p}: dst={ds} src={ss}" for p, ds, ss in mismatched[:20])
        raise ValueError(f"Weight shape mismatches while copying by path (showing up to 20):\n{msg}")
    if missing:
        # Some layers may be absent due to config differences; warn but continue.
        print(f"WARNING: {len(missing)} dst weights not found in src (showing up to 20):")
        for p in missing[:20]:
            print(f"  - {p}")


def main():
    args = parse_args()

    ckpt = Path(args.checkpoint)
    cfg = find_config(ckpt)
    weights_path = find_weights(ckpt)

    backbone = cfg.get("backbone", "mobilenetv2")
    alpha = cfg.get("alpha", 0.35)
    backbone_minimalistic = bool(cfg.get("backbone_minimalistic", False))
    backbone_include_preprocessing = bool(cfg.get("backbone_include_preprocessing", False))
    fpn_ch = int(cfg.get("fpn_ch", 32))
    simcc_ch = int(cfg.get("simcc_ch", 96))
    img_size = int(cfg.get("img_size", 256))
    num_bins = int(cfg.get("num_bins", img_size))
    tau = float(cfg.get("tau", 1.0))
    model_input_norm = str(cfg.get("input_norm", "imagenet")).lower().strip()

    tflite_input_norm = args.tflite_input_norm
    if tflite_input_norm == "auto":
        tflite_input_norm = model_input_norm
    tflite_input_norm = str(tflite_input_norm).lower().strip()

    print("=" * 60)
    print("Export int8 TFLite (PTQ)")
    print("=" * 60)
    print(f"Checkpoint: {ckpt}")
    print(f"Weights:    {weights_path}")
    print("Model config:")
    print(f"  backbone={backbone} alpha={alpha}")
    print(f"  img_size={img_size} num_bins={num_bins}")
    print(f"  fpn_ch={fpn_ch} simcc_ch={simcc_ch} tau={tau}")
    print(f"  model_input_norm={model_input_norm}")
    print(f"  tflite_input_norm={tflite_input_norm}")
    print(f"  quantization={args.quantization}")
    print(f"  io_dtype={args.io_dtype} output_dtype={args.output_dtype}")
    print(f"  allow_float_fallback={args.allow_float_fallback}")
    print(
        f"  axis_mean_impl={args.axis_mean_impl} global_pool_impl={args.global_pool_impl} score_pool_impl={args.score_pool_impl}"
    )
    print(f"  simcc_packed_layout={args.simcc_packed_layout}")

    # MobileNetV3 INT8: TFLite lowers hard-swish to HARD_SWISH, which XNNPACK does not
    # delegate in quantized graphs. Use a numerically-equivalent implementation that
    # avoids lowering to the HARD_SWISH builtin op.
    backbone_key = str(backbone).lower().strip().replace("-", "_")
    backbone_nonfused_hardswish = backbone_key.startswith("mobilenetv3") and str(args.quantization).lower() == "int8"
    if backbone_nonfused_hardswish:
        print("  backbone_nonfused_hardswish=True (XNNPACK int8 full-delegate workaround)")

    # Build + load model (avoid downloading backbone weights).
    # NOTE: we load weights into the "standard" model first for compatibility with checkpoints,
    # then copy them into an XNNPACK-friendly variant that avoids EXPAND_DIMS patterns from Conv1D.
    base_model = create_model(
        backbone=backbone,
        alpha=alpha,
        backbone_minimalistic=backbone_minimalistic,
        backbone_include_preprocessing=backbone_include_preprocessing,
        backbone_weights=None,
        fpn_ch=fpn_ch,
        simcc_ch=simcc_ch,
        img_size=img_size,
        num_bins=num_bins,
        tau=tau,
    )
    base_model.load_weights(str(weights_path))
    print(f"Loaded weights. Params: {base_model.count_params():,}")

    export_model = create_model(
        backbone=backbone,
        alpha=alpha,
        backbone_minimalistic=backbone_minimalistic,
        backbone_include_preprocessing=backbone_include_preprocessing,
        backbone_nonfused_hardswish=backbone_nonfused_hardswish,
        backbone_weights=None,
        fpn_ch=fpn_ch,
        simcc_ch=simcc_ch,
        img_size=img_size,
        num_bins=num_bins,
        tau=tau,
        conv1d_as_conv2d=True,
        axis_mean_impl=str(args.axis_mean_impl),
        global_pool_impl=str(args.global_pool_impl),
        score_pool_impl=None if str(args.score_pool_impl).lower().strip() == "auto" else str(args.score_pool_impl),
        simcc_output_layout=(
            "bins_first"
            if (str(args.output_format).lower().strip() == "simcc_logits")
            and (str(args.simcc_packed_layout).lower().strip() == "bins_first")
            else "corners_first"
        ),
    )
    _copy_weights_by_path(base_model, export_model)

    infer_model = create_tflite_inference_model(
        model=export_model,
        img_size=img_size,
        model_input_norm=model_input_norm,
        tflite_input_norm=tflite_input_norm,
        output_format=args.output_format,
        simcc_packed_layout=str(args.simcc_packed_layout),
    )

    # Convert
    converter = tf.lite.TFLiteConverter.from_keras_model(infer_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if args.quantization == "dynamic":
        # Weights-only quantization. No representative dataset required.
        pass
    else:
        # Representative dataset (calibration)
        ds = create_dataset(
            data_root=args.data_root,
            split=args.split,
            img_size=img_size,
            batch_size=1,
            shuffle=True,
            augment=False,
            drop_remainder=False,
            image_norm=tflite_input_norm,
        ).take(int(args.num_calib))

        def representative_dataset():
            for images, _targets in ds:
                # images: [1,H,W,3] float32 in the domain expected by infer_model input.
                yield [images]

        converter.representative_dataset = representative_dataset

    if args.quantization == "int16x8":
        int_ops = tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    else:
        int_ops = tf.lite.OpsSet.TFLITE_BUILTINS_INT8

    if args.quantization == "dynamic":
        # Let the converter choose (float builtins, with weights compressed).
        pass
    elif args.allow_float_fallback:
        converter.target_spec.supported_ops = [int_ops, tf.lite.OpsSet.TFLITE_BUILTINS]
    else:
        converter.target_spec.supported_ops = [int_ops]

    if args.quantization == "dynamic":
        # Keep float32 I/O for dynamic range quantization.
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    else:
        converter.inference_input_type = _io_dtype(args.io_dtype)
        converter.inference_output_type = _io_dtype(args.output_dtype)

    # Conversion tends to be more stable with this disabled for non-RNN models.
    converter._experimental_lower_tensor_list_ops = False  # pylint: disable=protected-access

    print("\nConverting...")
    tflite_model = converter.convert()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    print(f"Saved: {output_path} ({len(tflite_model) / (1024 * 1024):.2f} MB)")

    # Quick verification.
    print("\nVerifying...")
    interpreter = tf.lite.Interpreter(model_path=str(output_path), num_threads=int(args.threads))
    interpreter.allocate_tensors()
    in0 = interpreter.get_input_details()[0]
    outs = interpreter.get_output_details()
    print(f"  Input:  {in0['shape']} {in0['dtype']} quant={in0.get('quantization')}")
    for i, out in enumerate(outs):
        print(f"  Output {i}: {out['shape']} {out['dtype']} quant={out.get('quantization')} name={out.get('name')}")

    dummy = np.zeros(in0["shape"], dtype=in0["dtype"])
    interpreter.set_tensor(in0["index"], dummy)
    interpreter.invoke()
    for i, out in enumerate(outs):
        y = interpreter.get_tensor(out["index"])
        print(f"  Ran 1 inference. Output[{i}] shape: {np.asarray(y).shape}")


if __name__ == "__main__":
    main()
