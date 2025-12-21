"""
Evaluate DocCornerNetV3 TFLite models on dataset splits.

Goal: match `evaluate.py` metrics while running inference via TFLite Interpreter.
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
        description="Evaluate DocCornerNetV3 TFLite models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tflite_models",
        type=str,
        nargs="+",
        required=True,
        help="Paths to TFLite models to evaluate",
    )
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Dataset root (contains images/, labels/, split files, images-negative/)",
    )
    p.add_argument("--split", type=str, default="val", help="Split name (supports custom split files like val_cleaned)")
    p.add_argument(
        "--input_norm",
        type=str,
        default="imagenet",
        choices=["imagenet", "zero_one", "raw255"],
        help="Normalization applied BEFORE feeding the TFLite model. "
        "If your TFLite model includes preprocessing, use raw255.",
    )
    p.add_argument("--batch_size", type=int, default=32, help="tf.data batch size (TFLite runs per-sample)")
    p.add_argument("--num_samples", type=int, default=None, help="Limit number of evaluated samples")
    p.add_argument("--threads", type=int, default=1, help="TFLite interpreter CPU threads")
    p.add_argument("--warmup_runs", type=int, default=10, help="Warmup invokes before timing")
    p.add_argument(
        "--benchmark_runs",
        type=int,
        default=200,
        help="Max number of samples used to report latency stats",
    )
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


def _parse_outputs(interpreter: tf.lite.Interpreter) -> tuple[np.ndarray, float]:
    outs = interpreter.get_output_details()
    if not outs:
        raise RuntimeError("TFLite model has no outputs.")

    if len(outs) == 1:
        y = interpreter.get_tensor(outs[0]["index"])
        y = np.asarray(y)
        if y.ndim != 2 or y.shape[0] != 1:
            raise RuntimeError(f"Unexpected output shape: {y.shape}")
        if y.shape[1] == 9:
            coords = y[0, :8].astype(np.float32)
            score = float(y[0, 8])
        elif y.shape[1] == 8:
            coords = y[0, :8].astype(np.float32)
            score = 1.0
        else:
            raise RuntimeError(f"Unexpected output shape: {y.shape} (expected [1,9] or [1,8])")
        return np.clip(coords, 0.0, 1.0), float(np.clip(score, 0.0, 1.0))

    # Best-effort for legacy exports with two outputs (coords + score).
    y0 = interpreter.get_tensor(outs[0]["index"])
    y1 = interpreter.get_tensor(outs[1]["index"])
    coords = np.asarray(y0).reshape(-1).astype(np.float32)[:8]
    score = float(np.asarray(y1).reshape(-1)[0])
    return np.clip(coords, 0.0, 1.0), float(np.clip(score, 0.0, 1.0))


def _run_one(interpreter: tf.lite.Interpreter, x_nhwc: np.ndarray) -> tuple[np.ndarray, float, float]:
    input_detail = interpreter.get_input_details()[0]
    input_index = input_detail["index"]

    x = _quantize_if_needed(x_nhwc, input_detail)
    interpreter.set_tensor(input_index, x)
    t0 = time.perf_counter()
    interpreter.invoke()
    t1 = time.perf_counter()
    coords, score = _parse_outputs(interpreter)
    return coords, score, (t1 - t0) * 1000.0


def evaluate_tflite_model(
    tflite_path: str,
    data_root: str,
    split: str,
    input_norm: str,
    batch_size: int,
    num_samples: int | None,
    threads: int,
    warmup_runs: int,
    benchmark_runs: int,
) -> dict:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=threads)
    interpreter.allocate_tensors()

    img_size = _infer_img_size(interpreter)
    input_dtype = interpreter.get_input_details()[0]["dtype"]

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

    # Warmup on zeros (avoid measuring first-time allocations/compilation).
    in_detail = interpreter.get_input_details()[0]
    dummy = np.zeros(in_detail["shape"], dtype=in_detail["dtype"])
    for _ in range(max(0, int(warmup_runs))):
        interpreter.set_tensor(in_detail["index"], dummy)
        interpreter.invoke()

    seen = 0
    for images, targets in tqdm(dataset, desc=f"TFLite eval ({Path(tflite_path).name})"):
        x = images.numpy()
        coords_gt = targets["coords"].numpy()
        has_doc = targets["has_doc"].numpy()

        b = x.shape[0]
        coords_pred = np.zeros((b, 8), dtype=np.float32)
        score_pred = np.zeros((b,), dtype=np.float32)

        for j in range(b):
            coords, score, ms = _run_one(interpreter, x[j : j + 1])
            coords_pred[j] = coords
            score_pred[j] = score
            if len(lat_ms) < int(benchmark_runs):
                lat_ms.append(float(ms))

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
            "tflite_threads": int(threads),
            "tflite_input_dtype": str(np.dtype(input_dtype)),
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
    print("DocCornerNetV3 TFLite Evaluation")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Input normalization (fed to TFLite): {args.input_norm}")

    all_results = []
    for model_path in args.tflite_models:
        r = evaluate_tflite_model(
            tflite_path=model_path,
            data_root=args.data_root,
            split=args.split,
            input_norm=args.input_norm,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            threads=args.threads,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
        )
        all_results.append(r)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in all_results:
        print(f"\nModel: {r['model']}")
        print(f"  Mean IoU:          {r['mean_iou']:.4f}")
        print(f"  Corner Err (mean): {r['corner_error_px']:.2f}px (p95={r['corner_error_p95_px']:.2f}px)")
        print(f"  Recall@95:         {r['recall_95']*100:.1f}%")
        print(f"  Cls acc:           {r['cls_accuracy']*100:.1f}% (F1={r['cls_f1']*100:.1f}%)")
        print(f"  Latency p50/p95:   {r['latency_ms_p50']:.2f}/{r['latency_ms_p95']:.2f} ms (n={r['latency_ms_n']})")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
