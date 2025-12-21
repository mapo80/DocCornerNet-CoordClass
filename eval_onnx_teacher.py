"""
Evaluate ONNX heatmap teachers on DocCornerNet dataset splits.

The FastViT teachers in `teacher/` output a 4-channel heatmap:
  input:  img [B, 3, H, W] float32
  output: heatmap [B, 4, h, w] float32 (often after sigmoid)

We decode each channel to a corner point either with:
  - soft-argmax (expected value of heatmap marginals)
  - argmax (peak)

Metrics are computed only on positive samples (has_doc == 1), matching evaluate.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from dataset import create_dataset
from metrics import compute_corner_error, compute_polygon_iou


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate ONNX heatmap teachers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--onnx_model", type=str, required=True, help="Path to ONNX model")
    p.add_argument("--data_root", type=str, required=True, help="Dataset root")
    p.add_argument("--split", type=str, default="val", help="Split name (supports custom split files like val_cleaned)")
    p.add_argument("--img_size", type=int, default=256, help="Input image size (dataset resize)")
    p.add_argument(
        "--teacher_input_norm",
        type=str,
        default="zero_one",
        choices=["imagenet", "zero_one", "raw255", "m1p1"],
        help="Normalization expected by the ONNX model input.",
    )
    p.add_argument(
        "--decode",
        type=str,
        default="soft",
        choices=["soft", "argmax", "both"],
        help="Heatmap->coords decoding method",
    )
    p.add_argument("--batch_size", type=int, default=16, help="ONNX batch size")
    p.add_argument("--max_samples", type=int, default=0, help="If >0, stop after N positives")
    p.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="onnxruntime provider selection (best-effort)",
    )
    return p.parse_args()


def _convert_to_teacher_input(images_raw255_nhwc: np.ndarray, teacher_input_norm: str) -> np.ndarray:
    norm = teacher_input_norm.strip().lower()
    x = images_raw255_nhwc.astype(np.float32, copy=False)
    if norm == "raw255":
        pass
    elif norm == "zero_one":
        x = x / 255.0
    elif norm == "imagenet":
        x = x / 255.0
        x = (x - IMAGENET_MEAN[None, None, None, :]) / IMAGENET_STD[None, None, None, :]
    elif norm == "m1p1":
        x = (x / 127.5) - 1.0
    else:
        raise ValueError(f"Unsupported teacher_input_norm='{teacher_input_norm}'")

    # NCHW
    return np.transpose(x, (0, 3, 1, 2))


def _decode_soft(heat: np.ndarray) -> np.ndarray:
    """
    heat: [B, 4, h, w]
    returns coords: [B, 8] in [0,1]
    """
    b, c, h, w = heat.shape
    if c != 4:
        raise ValueError(f"Expected 4 channels, got {c}")

    eps = 1e-9
    # Marginals
    x_m = heat.sum(axis=2)  # [B,4,w]
    y_m = heat.sum(axis=3)  # [B,4,h]

    xs = np.arange(w, dtype=np.float32)[None, None, :]
    ys = np.arange(h, dtype=np.float32)[None, None, :]

    x_sum = x_m.sum(axis=-1, keepdims=True) + eps  # [B,4,1]
    y_sum = y_m.sum(axis=-1, keepdims=True) + eps

    x_px = (x_m * xs).sum(axis=-1) / x_sum.squeeze(-1)  # [B,4]
    y_px = (y_m * ys).sum(axis=-1) / y_sum.squeeze(-1)  # [B,4]

    x01 = x_px / float(max(w - 1, 1))
    y01 = y_px / float(max(h - 1, 1))

    coords = np.stack([x01, y01], axis=-1).reshape(b, 8).astype(np.float32)
    return np.clip(coords, 0.0, 1.0)


def _decode_argmax(heat: np.ndarray) -> np.ndarray:
    """
    heat: [B, 4, h, w]
    returns coords: [B, 8] in [0,1]
    """
    b, c, h, w = heat.shape
    if c != 4:
        raise ValueError(f"Expected 4 channels, got {c}")
    coords = np.empty((b, 8), dtype=np.float32)
    for bi in range(b):
        out = []
        for ci in range(4):
            idx = int(np.argmax(heat[bi, ci]))
            y = idx // w
            x = idx % w
            out.extend([x / float(max(w - 1, 1)), y / float(max(h - 1, 1))])
        coords[bi] = np.array(out, dtype=np.float32)
    return np.clip(coords, 0.0, 1.0)


def _order_quad_xy(pts_xy: np.ndarray) -> np.ndarray:
    """
    Reorder 4 points to (TL, TR, BR, BL) using sum/diff heuristic.
    Works well for most convex quads.
    """
    s = pts_xy.sum(axis=1)
    d = pts_xy[:, 0] - pts_xy[:, 1]
    tl = pts_xy[int(np.argmin(s))]
    br = pts_xy[int(np.argmax(s))]
    tr = pts_xy[int(np.argmax(d))]
    bl = pts_xy[int(np.argmin(d))]
    return np.stack([tl, tr, br, bl], axis=0)


def _order_quads(coords_8: np.ndarray) -> np.ndarray:
    """coords_8: [B,8] -> reordered [B,8] (TL,TR,BR,BL)."""
    b = coords_8.shape[0]
    out = np.empty_like(coords_8, dtype=np.float32)
    for i in range(b):
        pts = coords_8[i].reshape(4, 2)
        out[i] = _order_quad_xy(pts).reshape(-1)
    return out


def _summarize(name: str, ious: np.ndarray, errs: np.ndarray) -> dict:
    return {
        "name": name,
        "n": int(len(ious)),
        "mean_iou": float(np.mean(ious)) if len(ious) else 0.0,
        "median_iou": float(np.median(ious)) if len(ious) else 0.0,
        "p01_iou": float(np.percentile(ious, 1)) if len(ious) else 0.0,
        "p05_iou": float(np.percentile(ious, 5)) if len(ious) else 0.0,
        "frac_lt_50": float(np.mean(ious < 0.50)) if len(ious) else 0.0,
        "frac_lt_10": float(np.mean(ious < 0.10)) if len(ious) else 0.0,
        "recall_90": float(np.mean(ious >= 0.90)) if len(ious) else 0.0,
        "recall_95": float(np.mean(ious >= 0.95)) if len(ious) else 0.0,
        "corner_err_mean_px": float(np.mean(errs)) if len(errs) else 0.0,
        "corner_err_p95_px": float(np.percentile(errs, 95)) if len(errs) else 0.0,
    }


def main():
    args = parse_args()

    try:
        import onnxruntime as ort
    except Exception as e:
        raise SystemExit("onnxruntime is required. Install it in your env.") from e

    if args.provider == "cpu":
        providers = ["CPUExecutionProvider"]
    elif args.provider == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    sess = ort.InferenceSession(str(args.onnx_model), providers=providers)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    print("=" * 60)
    print("ONNX Teacher Evaluation")
    print("=" * 60)
    print(f"Model: {args.onnx_model}")
    print(f"Provider(s): {providers}")
    print(f"Split: {args.split}")
    print(f"Dataset img_size: {args.img_size}")
    print(f"Teacher input_norm: {args.teacher_input_norm}")
    print(f"Decode: {args.decode}")
    print()

    ds = create_dataset(
        data_root=args.data_root,
        split=args.split,
        img_size=args.img_size,
        batch_size=1,  # we batch ourselves after filtering positives (simpler)
        shuffle=False,
        augment=False,
        drop_remainder=False,
        image_norm="raw255",
    )

    # Accumulators for positives
    raw_imgs = []
    gts = []
    pos_seen = 0

    def flush(batch_raw, batch_gt):
        x = _convert_to_teacher_input(np.stack(batch_raw, axis=0), args.teacher_input_norm)
        t0 = time.perf_counter()
        heat = sess.run([out_name], {in_name: x})[0]
        t1 = time.perf_counter()
        return heat, (t1 - t0) * 1000.0

    ious_soft = []
    errs_soft = []
    ious_arg = []
    errs_arg = []
    infer_times = []

    for images, targets in ds:
        has_doc = float(targets["has_doc"].numpy().reshape(-1)[0])
        if has_doc < 0.5:
            continue

        raw_imgs.append(images.numpy()[0])  # [H,W,3] raw255
        gts.append(targets["coords"].numpy().reshape(-1))

        if len(raw_imgs) >= args.batch_size:
            heat, ms = flush(raw_imgs, gts)
            infer_times.append(ms / float(len(raw_imgs)))

            gt_arr = np.stack(gts, axis=0)
            if args.decode in {"soft", "both"}:
                pred = _order_quads(_decode_soft(heat))
                for p, gt in zip(pred, gt_arr):
                    ious_soft.append(compute_polygon_iou(p, gt))
                    errs_soft.append(compute_corner_error(p, gt, img_size=args.img_size)[0])
            if args.decode in {"argmax", "both"}:
                pred = _order_quads(_decode_argmax(heat))
                for p, gt in zip(pred, gt_arr):
                    ious_arg.append(compute_polygon_iou(p, gt))
                    errs_arg.append(compute_corner_error(p, gt, img_size=args.img_size)[0])

            pos_seen += len(raw_imgs)
            raw_imgs, gts = [], []
            if args.max_samples and pos_seen >= args.max_samples:
                break

    # Flush remainder
    if raw_imgs:
        heat, ms = flush(raw_imgs, gts)
        infer_times.append(ms / float(len(raw_imgs)))
        gt_arr = np.stack(gts, axis=0)
        if args.decode in {"soft", "both"}:
            pred = _order_quads(_decode_soft(heat))
            for p, gt in zip(pred, gt_arr):
                ious_soft.append(compute_polygon_iou(p, gt))
                errs_soft.append(compute_corner_error(p, gt, img_size=args.img_size)[0])
        if args.decode in {"argmax", "both"}:
            pred = _order_quads(_decode_argmax(heat))
            for p, gt in zip(pred, gt_arr):
                ious_arg.append(compute_polygon_iou(p, gt))
                errs_arg.append(compute_corner_error(p, gt, img_size=args.img_size)[0])

        pos_seen += len(raw_imgs)

    print(f"Positives evaluated: {pos_seen}")
    if infer_times:
        per = np.array(infer_times, dtype=np.float32)
        print(f"ONNX infer (per-sample): mean={float(per.mean()):.2f} ms  p95={float(np.percentile(per,95)):.2f} ms")
    print()

    if args.decode in {"soft", "both"}:
        r = _summarize("soft", np.array(ious_soft, dtype=np.float32), np.array(errs_soft, dtype=np.float32))
        print("Decode=soft")
        print(f"  Mean IoU:   {r['mean_iou']:.4f}")
        print(f"  Median IoU: {r['median_iou']:.4f}")
        print(f"  P01/P05:    {r['p01_iou']:.4f} / {r['p05_iou']:.4f}")
        print(f"  IoU<0.50:   {r['frac_lt_50']*100:.2f}%  (IoU<0.10: {r['frac_lt_10']*100:.2f}%)")
        print(f"  R@95:       {r['recall_95']*100:.2f}%")
        print(f"  Err mean:   {r['corner_err_mean_px']:.2f}px (p95={r['corner_err_p95_px']:.2f}px)")
        print()

    if args.decode in {"argmax", "both"}:
        r = _summarize("argmax", np.array(ious_arg, dtype=np.float32), np.array(errs_arg, dtype=np.float32))
        print("Decode=argmax")
        print(f"  Mean IoU:   {r['mean_iou']:.4f}")
        print(f"  Median IoU: {r['median_iou']:.4f}")
        print(f"  P01/P05:    {r['p01_iou']:.4f} / {r['p05_iou']:.4f}")
        print(f"  IoU<0.50:   {r['frac_lt_50']*100:.2f}%  (IoU<0.10: {r['frac_lt_10']*100:.2f}%)")
        print(f"  R@95:       {r['recall_95']*100:.2f}%")
        print(f"  Err mean:   {r['corner_err_mean_px']:.2f}px (p95={r['corner_err_p95_px']:.2f}px)")
        print()


if __name__ == "__main__":
    main()
