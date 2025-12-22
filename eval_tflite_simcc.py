"""
Evaluate a TFLite model that outputs SimCC logits (simcc_x, simcc_y, score_logit).

This is useful for fully-quantized int8 exports where decoding coordinates inside the
TFLite graph is too sensitive. We keep the model fully int8 (with float32 outputs)
and decode in float32 outside the model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dataset import create_dataset
from metrics import ValidationMetrics


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate SimCC-logits TFLite models (decode outside the model)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tflite_models", type=str, nargs="+", required=True, help="Paths to TFLite models")
    p.add_argument("--data_root", type=str, required=True, help="Dataset root")
    p.add_argument("--split", type=str, default="val_cleaned", help="Split name")
    p.add_argument(
        "--input_norm",
        type=str,
        default="imagenet",
        choices=["imagenet", "zero_one", "raw255"],
        help="Normalization applied BEFORE feeding the TFLite model",
    )
    p.add_argument("--tau", type=float, default=1.0, help="Softmax temperature used for SimCC decode")
    p.add_argument("--batch_size", type=int, default=32, help="tf.data batch size (TFLite runs per-sample)")
    p.add_argument("--num_samples", type=int, default=None, help="Limit number of evaluated samples")
    p.add_argument("--threads", type=int, default=1, help="TFLite interpreter CPU threads")
    p.add_argument("--warmup_runs", type=int, default=10, help="Warmup invokes before timing")
    p.add_argument("--benchmark_runs", type=int, default=200, help="Max number of samples used to report latency")
    p.add_argument("--output", type=str, default=None, help="Write results JSON to this path")
    return p.parse_args()


def _infer_img_size(interpreter: tf.lite.Interpreter) -> int:
    details = interpreter.get_input_details()
    if not details:
        raise RuntimeError("TFLite model has no inputs.")
    shape = details[0]["shape"]
    if len(shape) != 4:
        raise RuntimeError(f"Unexpected input shape {shape} (expected [B,H,W,C])")
    h = int(shape[1])
    w = int(shape[2])
    c = int(shape[3])
    if c != 3:
        raise RuntimeError(f"Unexpected input channels C={c} (expected 3)")
    if h != w:
        raise RuntimeError(f"Non-square input not supported: H={h}, W={w}")
    return h


def _quantize_if_needed(x: np.ndarray, input_detail: dict) -> np.ndarray:
    dtype = input_detail["dtype"]
    if dtype not in (np.uint8, np.int8):
        return x.astype(dtype, copy=False)

    scale, zero_point = input_detail.get("quantization", (0.0, 0))
    if not scale:
        raise RuntimeError("Quantized input without quantization parameters.")

    xq = np.round(x / scale + zero_point)
    if dtype == np.uint8:
        xq = np.clip(xq, 0, 255)
    else:
        xq = np.clip(xq, -128, 127)
    return xq.astype(dtype)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _dequantize_if_needed(x: np.ndarray, detail: dict) -> np.ndarray:
    dtype = np.asarray(x).dtype
    if dtype not in (np.uint8, np.int8):
        return np.asarray(x, dtype=np.float32)
    scale, zero_point = detail.get("quantization", (0.0, 0))
    if not scale:
        raise RuntimeError("Quantized output without quantization parameters.")
    return (np.asarray(x).astype(np.float32) - float(zero_point)) * float(scale)


def _decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray, tau: float) -> np.ndarray:
    """
    Decode SimCC logits to normalized coords.

    Args:
      simcc_x: [1,4,num_bins]
      simcc_y: [1,4,num_bins]
    Returns:
      coords: [8] (x0,y0,x1,y1,x2,y2,x3,y3) in [0,1]
    """
    if simcc_x.ndim != 3 or simcc_y.ndim != 3:
        raise RuntimeError(f"Unexpected simcc shapes: x={simcc_x.shape}, y={simcc_y.shape} (expected [1,4,num_bins])")
    if simcc_x.shape[0] != 1 or simcc_y.shape[0] != 1 or simcc_x.shape[1] != 4 or simcc_y.shape[1] != 4:
        raise RuntimeError(f"Unexpected simcc shapes: x={simcc_x.shape}, y={simcc_y.shape}")
    if simcc_x.shape[2] != simcc_y.shape[2]:
        raise RuntimeError(f"Mismatched num_bins: x={simcc_x.shape[2]} y={simcc_y.shape[2]}")

    num_bins = int(simcc_x.shape[2])
    centers = np.linspace(0.0, 1.0, num_bins, dtype=np.float32).reshape(1, 1, num_bins)

    # Softmax (stable)
    x = simcc_x.astype(np.float32) / float(tau)
    y = simcc_y.astype(np.float32) / float(tau)

    x = x - np.max(x, axis=-1, keepdims=True)
    y = y - np.max(y, axis=-1, keepdims=True)

    px = np.exp(x)
    py = np.exp(y)
    px = px / (np.sum(px, axis=-1, keepdims=True) + 1e-8)
    py = py / (np.sum(py, axis=-1, keepdims=True) + 1e-8)

    # Expectation -> [1,4]
    ex = np.sum(px * centers, axis=-1).reshape(4)
    ey = np.sum(py * centers, axis=-1).reshape(4)

    coords = np.stack([ex, ey], axis=-1).reshape(8)
    return np.clip(coords, 0.0, 1.0).astype(np.float32)


def _decode_simcc_packed(simcc_xy: np.ndarray, tau: float) -> np.ndarray:
    """
    Decode packed SimCC logits tensor to normalized coords.

    Args:
      simcc_xy: packed logits, either:
        - [1,8,num_bins] where first 4 are X, next 4 are Y (8-first)
        - [1,num_bins,8] where last dim packs [x0..x3,y0..y3] (bins-first)
    Returns:
      coords: [8] (x0,y0,x1,y1,x2,y2,x3,y3) in [0,1]
    """
    if simcc_xy.ndim != 3:
        raise RuntimeError(f"Unexpected simcc_xy shape: {simcc_xy.shape} (expected [1,8,num_bins])")
    if simcc_xy.shape[0] != 1:
        raise RuntimeError(f"Unexpected simcc_xy shape: {simcc_xy.shape} (expected batch=1)")

    # Layout 1: [1,8,num_bins]
    if simcc_xy.shape[1] == 8:
        simcc_x = simcc_xy[:, :4, :]
        simcc_y = simcc_xy[:, 4:, :]
        return _decode_simcc(simcc_x, simcc_y, tau=tau)

    # Layout 2: [1,num_bins,8]
    if simcc_xy.shape[2] == 8:
        simcc_x_l = simcc_xy[:, :, :4]  # [1,num_bins,4]
        simcc_y_l = simcc_xy[:, :, 4:]  # [1,num_bins,4]
        simcc_x = np.transpose(simcc_x_l, (0, 2, 1))  # [1,4,num_bins]
        simcc_y = np.transpose(simcc_y_l, (0, 2, 1))  # [1,4,num_bins]
        return _decode_simcc(simcc_x, simcc_y, tau=tau)

    raise RuntimeError(
        f"Unexpected simcc_xy shape: {simcc_xy.shape} (expected [1,8,num_bins] or [1,num_bins,8])"
    )


def evaluate_model(
    tflite_path: str,
    data_root: str,
    split: str,
    input_norm: str,
    tau: float,
    batch_size: int,
    num_samples: int | None,
    threads: int,
    warmup_runs: int,
    benchmark_runs: int,
) -> dict:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=int(threads))
    interpreter.allocate_tensors()

    img_size = _infer_img_size(interpreter)
    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    if len(output_details) not in (2, 3):
        raise RuntimeError(f"Expected 2 outputs [simcc_xy, score_logit] or 3 outputs, got {len(output_details)} outputs.")

    def _get_out_by_rank(rank: int):
        for out in output_details:
            shape = out.get("shape")
            if shape is None:
                continue
            if len(shape) == rank:
                return out
        return None

    dataset = create_dataset(
        data_root=data_root,
        split=split,
        img_size=img_size,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        augment=False,
        drop_remainder=False,
        image_norm=input_norm,
    )
    if num_samples:
        dataset = dataset.take(int(np.ceil(num_samples / max(1, int(batch_size)))))

    metrics = ValidationMetrics(img_size=img_size)
    lat_ms: list[float] = []

    # Warmup
    dummy = np.zeros(input_detail["shape"], dtype=input_detail["dtype"])
    for _ in range(max(0, int(warmup_runs))):
        interpreter.set_tensor(input_detail["index"], dummy)
        interpreter.invoke()

    seen = 0
    for images, targets in tqdm(dataset, desc=f"TFLite eval (SimCC logits) ({Path(tflite_path).name})"):
        x = images.numpy()
        coords_gt = targets["coords"].numpy()
        has_doc = targets["has_doc"].numpy()

        b = x.shape[0]
        coords_pred = np.zeros((b, 8), dtype=np.float32)
        score_pred = np.zeros((b,), dtype=np.float32)

        for j in range(b):
            x1 = _quantize_if_needed(x[j : j + 1], input_detail)
            interpreter.set_tensor(input_detail["index"], x1)
            t0 = time.perf_counter()
            interpreter.invoke()
            t1 = time.perf_counter()

            if len(output_details) == 2:
                # Order is not guaranteed; disambiguate by shape.
                out_a = output_details[0]
                out_b = output_details[1]
                a_shape = tuple(int(x) for x in out_a.get("shape", []))
                b_shape = tuple(int(x) for x in out_b.get("shape", []))

                if len(a_shape) == 2 and a_shape[1] == 1:
                    score_out = out_a
                    simcc_out = out_b
                elif len(b_shape) == 2 and b_shape[1] == 1:
                    score_out = out_b
                    simcc_out = out_a
                else:
                    # Fallback: assume larger tensor is simcc.
                    score_out = out_a if np.prod(a_shape) <= np.prod(b_shape) else out_b
                    simcc_out = out_b if score_out is out_a else out_a

                simcc_xy_raw = interpreter.get_tensor(simcc_out["index"])
                score_logit_raw = interpreter.get_tensor(score_out["index"])

                simcc_xy = _dequantize_if_needed(simcc_xy_raw, simcc_out)
                score_logit = _dequantize_if_needed(score_logit_raw, score_out)

                coords_pred[j] = _decode_simcc_packed(simcc_xy, tau=float(tau))
            else:
                simcc_x_raw = interpreter.get_tensor(output_details[0]["index"])
                simcc_y_raw = interpreter.get_tensor(output_details[1]["index"])
                score_logit_raw = interpreter.get_tensor(output_details[2]["index"])

                simcc_x = _dequantize_if_needed(simcc_x_raw, output_details[0])
                simcc_y = _dequantize_if_needed(simcc_y_raw, output_details[1])
                score_logit = _dequantize_if_needed(score_logit_raw, output_details[2])

                coords_pred[j] = _decode_simcc(simcc_x, simcc_y, tau=float(tau))

            score_pred[j] = float(_sigmoid(np.asarray(score_logit).reshape(-1)[0]))

            if len(lat_ms) < int(benchmark_runs):
                lat_ms.append(float((t1 - t0) * 1000.0))

        metrics.update(coords_pred, coords_gt, score_pred, has_doc)

        seen += b
        if num_samples and seen >= num_samples:
            break

    results = metrics.compute()
    results.update(
        {
            "model": str(tflite_path),
            "img_size": int(img_size),
            "input_norm": str(input_norm),
            "tau": float(tau),
            "tflite_threads": int(threads),
            "tflite_input_dtype": str(np.dtype(input_detail["dtype"])),
        }
    )

    if lat_ms:
        arr = np.array(lat_ms, dtype=np.float32)
        results["latency_ms_p50"] = float(np.percentile(arr, 50))
        results["latency_ms_p95"] = float(np.percentile(arr, 95))
        results["latency_ms_mean"] = float(arr.mean())
        results["latency_ms_n"] = int(len(arr))
    else:
        results["latency_ms_p50"] = 0.0
        results["latency_ms_p95"] = 0.0
        results["latency_ms_mean"] = 0.0
        results["latency_ms_n"] = 0

    return results


def main():
    args = parse_args()

    print("=" * 60)
    print("DocCornerNetV3 TFLite Evaluation (SimCC logits)")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Input normalization (fed to TFLite): {args.input_norm}")
    print(f"Decode tau: {args.tau}")

    all_results = []
    for model_path in args.tflite_models:
        r = evaluate_model(
            tflite_path=model_path,
            data_root=args.data_root,
            split=args.split,
            input_norm=args.input_norm,
            tau=args.tau,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            threads=args.threads,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
        )
        all_results.append(r)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60 + "\n")

    for r in all_results:
        print(f"Model: {Path(r['model']).name}")
        print(f"  Mean IoU:          {r['mean_iou']:.4f}")
        print(f"  Corner Err (mean): {r['corner_error_px']:.2f}px (p95={r['corner_error_p95_px']:.2f}px)")
        print(f"  Recall@95:         {r['recall_95']*100:.1f}%")
        print(f"  Cls acc:           {r['cls_accuracy']*100:.1f}% (F1={r['cls_f1']*100:.1f}%)")
        print(f"  Latency p50/p95:   {r['latency_ms_p50']:.2f}/{r['latency_ms_p95']:.2f} ms (n={r['latency_ms_n']})")
        print()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_results, indent=2))
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
