"""
Benchmark TFLite inference latency on real dataset samples (CPU).

This script is meant for *latency only* (no accuracy metrics).
It measures:
  - invoke_ms: just `interpreter.invoke()` time (default)
  - io_ms: set_tensor + invoke + get_tensor (end-to-end inside Interpreter API)

Notes:
  - Image decode / tf.data preprocessing time is NOT included (we time only inside the Interpreter API).
  - For quantized inputs (int8/uint8), input quantization is done outside the timed region.
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark TFLite inference latency on dataset splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tflite_models", type=str, nargs="+", required=True, help="Paths to .tflite models")
    p.add_argument("--data_root", type=str, required=True, help="Dataset root (images/, labels/, split files, ...)")
    p.add_argument("--split", type=str, default="val", help="Split name (e.g. val, val_clean_iter4_mix)")
    p.add_argument(
        "--input_norm",
        type=str,
        default="imagenet",
        choices=["imagenet", "zero_one", "raw255"],
        help="Normalization applied BEFORE feeding the TFLite model.",
    )
    p.add_argument("--threads", type=int, default=4, help="TFLite interpreter threads")
    p.add_argument("--batch_size", type=int, default=32, help="tf.data batch size (TFLite runs per-sample)")
    p.add_argument("--num_samples", type=int, default=2000, help="Number of samples to time (<=0 means all)")
    p.add_argument("--warmup_runs", type=int, default=25, help="Warmup invokes before timing")
    p.add_argument(
        "--mode",
        type=str,
        default="invoke",
        choices=["invoke", "io"],
        help=(
            "'invoke' measures interpreter.invoke() only; "
            "'io' measures set_tensor + invoke + get_tensor."
        ),
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
    h, w, c = int(shape[1]), int(shape[2]), int(shape[3])
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


def benchmark_model(
    tflite_path: str,
    data_root: str,
    split: str,
    input_norm: str,
    threads: int,
    batch_size: int,
    num_samples: int,
    warmup_runs: int,
    mode: str,
) -> dict:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=int(threads))
    interpreter.allocate_tensors()

    img_size = _infer_img_size(interpreter)
    in_detail = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()

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

    # Warmup on zeros (avoid measuring first-time allocations/compilation).
    dummy = np.zeros(in_detail["shape"], dtype=in_detail["dtype"])
    for _ in range(max(0, int(warmup_runs))):
        interpreter.set_tensor(in_detail["index"], dummy)
        interpreter.invoke()
        if mode == "io":
            for od in out_details:
                _ = interpreter.get_tensor(od["index"])

    lat_ms: list[float] = []
    seen = 0

    target_n = None if int(num_samples) <= 0 else int(num_samples)

    for images, _targets in tqdm(dataset, desc=f"Bench ({Path(tflite_path).name})", leave=False):
        x = images.numpy()
        b = x.shape[0]
        for j in range(b):
            if target_n is not None and seen >= target_n:
                break
            x1 = _quantize_if_needed(x[j : j + 1], in_detail)
            if mode == "io":
                t0 = time.perf_counter()
                interpreter.set_tensor(in_detail["index"], x1)
                interpreter.invoke()
                for od in out_details:
                    _ = interpreter.get_tensor(od["index"])
                t1 = time.perf_counter()
            else:
                interpreter.set_tensor(in_detail["index"], x1)
                t0 = time.perf_counter()
                interpreter.invoke()
                t1 = time.perf_counter()
            lat_ms.append((t1 - t0) * 1000.0)
            seen += 1
        if target_n is not None and seen >= target_n:
            break

    if not lat_ms:
        raise RuntimeError("No samples benchmarked (check split / dataset path).")

    arr = np.asarray(lat_ms, dtype=np.float32)
    return {
        "model": str(tflite_path),
        "img_size": int(img_size),
        "tflite_threads": int(threads),
        "mode": str(mode),
        "input_norm": str(input_norm),
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "latency_ms_n": int(arr.size),
    }


def main():
    args = parse_args()

    results = []
    for model_path in args.tflite_models:
        r = benchmark_model(
            tflite_path=model_path,
            data_root=args.data_root,
            split=args.split,
            input_norm=args.input_norm,
            threads=args.threads,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            warmup_runs=args.warmup_runs,
            mode=str(args.mode).lower().strip(),
        )
        results.append(r)
        print(
            f"{Path(model_path).name}: mean={r['latency_ms_mean']:.3f}ms "
            f"p50={r['latency_ms_p50']:.3f}ms p95={r['latency_ms_p95']:.3f}ms "
            f"(n={r['latency_ms_n']}, img={r['img_size']}, threads={r['tflite_threads']}, mode={r['mode']})"
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "data_root": str(args.data_root),
                    "split": str(args.split),
                    "input_norm": str(args.input_norm),
                    "mode": str(args.mode),
                    "threads": int(args.threads),
                    "batch_size": int(args.batch_size),
                    "num_samples": int(args.num_samples),
                    "warmup_runs": int(args.warmup_runs),
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

