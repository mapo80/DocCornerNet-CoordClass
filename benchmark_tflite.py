"""
Microbenchmark TFLite inference latency (CPU).

Note: this measures interpreter overhead + invoke (and output read), not image decode/preprocess.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args():
    p = argparse.ArgumentParser(
        description="Microbenchmark a TFLite model (CPU)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", type=str, required=True, help="Path to .tflite file")
    p.add_argument("--threads", type=int, default=1, help="TFLite interpreter threads")
    p.add_argument("--warmup", type=int, default=50, help="Warmup invokes")
    p.add_argument("--runs", type=int, default=500, help="Timed invokes")
    p.add_argument(
        "--input_fill",
        type=str,
        default="random",
        choices=["zeros", "random"],
        help="Input tensor fill strategy",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for random inputs")
    return p.parse_args()


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


def main():
    args = parse_args()

    interpreter = tf.lite.Interpreter(model_path=str(args.model), num_threads=args.threads)
    interpreter.allocate_tensors()

    in_detail = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()

    shape = in_detail["shape"]
    dtype = in_detail["dtype"]

    rng = np.random.default_rng(args.seed)
    if args.input_fill == "zeros":
        x = np.zeros(shape, dtype=np.float32)
    else:
        # Use a bounded range that is reasonable for most float models; quantized inputs get quantized below.
        x = rng.random(shape, dtype=np.float32)

    x = _quantize_if_needed(x, in_detail)

    # Warmup
    for _ in range(max(0, args.warmup)):
        interpreter.set_tensor(in_detail["index"], x)
        interpreter.invoke()
        for od in out_details:
            _ = interpreter.get_tensor(od["index"])

    # Timed runs
    times = np.empty((max(0, args.runs),), dtype=np.float32)
    for i in range(times.shape[0]):
        t0 = time.perf_counter()
        interpreter.set_tensor(in_detail["index"], x)
        interpreter.invoke()
        for od in out_details:
            _ = interpreter.get_tensor(od["index"])
        t1 = time.perf_counter()
        times[i] = (t1 - t0) * 1000.0

    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))
    mean = float(times.mean())

    model_name = Path(args.model).name
    print(f"Model: {model_name}")
    print(f"Input: shape={tuple(shape)}, dtype={np.dtype(dtype)}")
    print(f"Threads: {args.threads}")
    print(f"Latency: p50={p50:.3f} ms  p95={p95:.3f} ms  mean={mean:.3f} ms  (n={len(times)})")


if __name__ == "__main__":
    main()

