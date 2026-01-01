#!/usr/bin/env python3
"""
Generate DocAligner HeatmapReg pseudo-labels (corners) for dataset splits using ONNXRuntime.

This is a lightweight alternative to importing the full DocAligner stack (capybara),
useful for running large-scale annotation on GPU pods.

Writes:
  - <data_root>/<split>_docaligner.txt

Each line has a fixed format (11 tokens):
  filename detected confidence x0 y0 x1 y1 x2 y2 x3 y3

Notes:
  - It preserves the split file order and includes negatives as detected=0 with coords=-1.
  - It does NOT modify labels/ nor train.txt/val.txt.
  - `confidence` is the mean of the 4 per-corner heatmap maxima (0..1) when detected=1.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _read_split_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text().splitlines():
        name = raw.strip()
        if not name:
            continue
        if name.startswith("images/"):
            name = name[len("images/") :]
        elif name.startswith("images-negative/"):
            name = name[len("images-negative/") :]
        lines.append(name)
    return lines


def _order_points_tl_tr_br_bl(coords8: np.ndarray) -> np.ndarray:
    pts = coords8.reshape(4, 2).astype(np.float32)
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts_ccw = pts[np.argsort(angles)]
    start = int(np.argmin(pts_ccw[:, 0] + pts_ccw[:, 1]))
    pts_ccw = np.roll(pts_ccw, -start, axis=0)
    if pts_ccw[1, 1] > pts_ccw[-1, 1]:
        pts_ccw = np.vstack([pts_ccw[0:1], pts_ccw[:0:-1]])
    return pts_ccw.reshape(-1)


def _resolve_image_path(data_root: Path, name: str) -> tuple[Path | None, bool]:
    p = data_root / "images" / name
    if p.exists():
        return p, False
    p2 = data_root / "images-negative" / name
    if p2.exists():
        return p2, True
    return None, name.startswith("negative_")


def _preprocess_to_chw_256(img: Image.Image) -> np.ndarray:
    # Matches DocAligner heatmap_reg preprocess: RGB -> resize 256 -> CHW float32 in [0..1]
    img = img.convert("RGB")
    img = img.resize((256, 256), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32)  # HWC, 0..255
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = x / 255.0
    return x


def _coords_from_heatmap(
    hm: np.ndarray,
    heatmap_threshold: float,
    w: int,
    h: int,
) -> tuple[bool, float, float, float]:
    """
    Returns (ok, maxv, x_norm, y_norm) using a weighted centroid over hm>=threshold.
    hm is 2D float32 (128x128), values in [0..1] (sigmoid already applied in ONNX).
    """
    if hm.ndim != 2:
        return False, 0.0, -1.0, -1.0
    maxv = float(np.max(hm))
    if not np.isfinite(maxv) or maxv < float(heatmap_threshold):
        return False, maxv, -1.0, -1.0

    mask = hm >= float(heatmap_threshold)
    if not np.any(mask):
        return False, maxv, -1.0, -1.0

    ys, xs = np.nonzero(mask)
    weights = hm[mask].astype(np.float64)
    wsum = float(weights.sum())
    if wsum <= 0.0 or not np.isfinite(wsum):
        return False, maxv, -1.0, -1.0

    x_hm = float((xs.astype(np.float64) * weights).sum() / wsum)
    y_hm = float((ys.astype(np.float64) * weights).sum() / wsum)

    # Map [0..(hw-1)] -> [0..(W-1)] then normalize by W/H (as in other scripts).
    hm_h, hm_w = int(hm.shape[0]), int(hm.shape[1])
    x_px = (x_hm / max(1.0, float(hm_w - 1))) * max(1.0, float(w - 1))
    y_px = (y_hm / max(1.0, float(hm_h - 1))) * max(1.0, float(h - 1))
    x_norm = float(np.clip(x_px / float(w), 0.0, 1.0))
    y_norm = float(np.clip(y_px / float(h), 0.0, 1.0))
    return True, maxv, x_norm, y_norm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate DocAligner heatmap ONNX pseudo-label split files (<split>_docaligner.txt).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new",
        help="Dataset root (contains images/, images-negative/, train.txt, val.txt).",
    )
    p.add_argument("--splits", type=str, default="train,val", help="Comma-separated split basenames")
    p.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="Path to fastvit_sa24_h_e_bifpn_256_fp32.onnx",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="ONNXRuntime execution device preference",
    )
    p.add_argument("--batch_size", type=int, default=16, help="Inference batch size (dynamic batch supported)")
    p.add_argument(
        "--heatmap_threshold",
        type=float,
        default=0.3,
        help="Heatmap threshold (matches DocAligner default)",
    )
    p.add_argument("--max_images", type=int, default=0, help="If >0, limit per split (debug)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing <split>_docaligner.txt")
    return p.parse_args()


@dataclass(frozen=True)
class _Meta:
    name: str
    w: int
    h: int


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]

    onnx_path = Path(args.onnx_path)
    if not onnx_path.exists():
        raise SystemExit(f"Missing ONNX: {onnx_path}")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import onnxruntime as ort  # lazy import

    providers: list[str]
    if str(args.device).lower() == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)
    input_name = session.get_inputs()[0].name

    print("\n" + "=" * 80)
    print("Generating DocAligner heatmap (ONNX) split files")
    print("=" * 80)
    print(f"Dataset: {data_root}")
    print(f"Splits: {splits}")
    print(f"ONNX: {onnx_path}")
    print(f"Providers: {session.get_providers()}")
    print(f"batch_size={args.batch_size} heatmap_threshold={args.heatmap_threshold}")

    for split in splits:
        split_file = data_root / f"{split}.txt"
        if not split_file.exists():
            raise SystemExit(f"Missing split file: {split_file}")

        out_path = data_root / f"{split}_docaligner.txt"
        if out_path.exists() and not bool(args.overwrite):
            raise SystemExit(f"Refusing to overwrite existing {out_path}. Use --overwrite.")

        names = _read_split_lines(split_file)
        if args.max_images and args.max_images > 0:
            names = names[: int(args.max_images)]

        print(f"\n{split}: {len(names)} entries -> {out_path.name}")

        batch_x: list[np.ndarray] = []
        batch_meta: list[_Meta] = []

        def flush(fh) -> None:
            if not batch_meta:
                return
            x = np.stack(batch_x, axis=0).astype(np.float32, copy=False)  # [B,3,256,256]
            heatmap = session.run(None, {input_name: x})[0]  # [B,4,128,128]
            heatmap = np.asarray(heatmap, dtype=np.float32)

            for i, meta in enumerate(batch_meta):
                pts: list[float] = []
                confs: list[float] = []
                ok_all = True
                for ch in range(4):
                    ok, maxv, x_n, y_n = _coords_from_heatmap(
                        heatmap[i, ch],
                        heatmap_threshold=float(args.heatmap_threshold),
                        w=int(meta.w),
                        h=int(meta.h),
                    )
                    if not ok:
                        ok_all = False
                        break
                    confs.append(float(maxv))
                    pts.extend([float(x_n), float(y_n)])

                if not ok_all or len(pts) != 8:
                    detected = 0
                    confidence = 0.0
                    coords = np.full((8,), -1.0, dtype=np.float32)
                else:
                    detected = 1
                    confidence = float(np.clip(np.mean(confs) if confs else 0.0, 0.0, 1.0))
                    coords = np.asarray(pts, dtype=np.float32)
                    coords = np.clip(coords, 0.0, 1.0)
                    coords = _order_points_tl_tr_br_bl(coords)

                parts = [meta.name, str(int(detected)), f"{float(confidence):.6f}"] + [
                    f"{float(v):.6f}" for v in coords.tolist()
                ]
                fh.write(" ".join(parts) + "\n")

            batch_x.clear()
            batch_meta.clear()

        with out_path.open("w") as f:
            for name in tqdm(names, desc=f"DocAligner-ONNX ({split})"):
                if name.startswith("negative_"):
                    parts = [name, "0", "0.000000"] + ["-1.000000"] * 8
                    f.write(" ".join(parts) + "\n")
                    continue

                img_path, is_neg_dir = _resolve_image_path(data_root, name)
                if is_neg_dir:
                    parts = [name, "0", "0.000000"] + ["-1.000000"] * 8
                    f.write(" ".join(parts) + "\n")
                    continue

                if img_path is None:
                    parts = [name, "0", "0.000000"] + ["-1.000000"] * 8
                    f.write(" ".join(parts) + "\n")
                    continue

                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        x = _preprocess_to_chw_256(img)
                except Exception:
                    parts = [name, "0", "0.000000"] + ["-1.000000"] * 8
                    f.write(" ".join(parts) + "\n")
                    continue

                batch_x.append(x)
                batch_meta.append(_Meta(name=name, w=int(w), h=int(h)))

                if len(batch_meta) >= int(args.batch_size):
                    flush(f)

            flush(f)

    print("\nDone.")


if __name__ == "__main__":
    main()

