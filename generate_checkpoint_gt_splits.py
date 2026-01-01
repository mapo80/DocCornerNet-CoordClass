#!/usr/bin/env python3
"""
Generate model pseudo-labels (corners + score) for dataset splits from a DocCornerNet checkpoint.

Writes:
  <data_root>/<split>_<suffix>.txt

Each line has a fixed format (11 tokens):
  filename detected score x0 y0 x1 y1 x2 y2 x3 y3

Notes:
  - It preserves the split file order and includes negatives as detected=0 with coords=-1.
  - It does NOT modify labels/ nor train.txt/val.txt.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

import tensorflow as tf

from model import load_inference_model


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
    # Ensure second point is "top" neighbor (TR) not bottom (BL)
    if pts_ccw[1, 1] > pts_ccw[-1, 1]:
        pts_ccw = np.vstack([pts_ccw[0:1], pts_ccw[:0:-1]])
    return pts_ccw.reshape(-1)


def _preprocess_image(img: Image.Image, img_size: int) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x


def _load_config(checkpoint_dir: Path) -> dict:
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    return {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate pseudo-label split files from a DocCornerNet checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new",
        help="Dataset root (contains images/, images-negative/, train.txt, val.txt).",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path("checkpoints") / "mobilenetv2_256_clean_iter3"),
        help="Checkpoint directory (contains config.json and best_model.weights.h5).",
    )
    p.add_argument("--splits", type=str, default="train,val", help="Comma-separated split basenames")
    p.add_argument("--suffix", type=str, default="winner", help="Output suffix for <split>_<suffix>.txt")
    p.add_argument("--batch_size", type=int, default=128, help="Inference batch size")
    p.add_argument("--score_threshold", type=float, default=0.5, help="Detection threshold for 'detected' flag")
    p.add_argument("--max_images", type=int, default=0, help="If >0, limit number of items per split (debug)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    checkpoint_dir = Path(args.checkpoint)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]

    if not (checkpoint_dir / "best_model.weights.h5").exists():
        raise SystemExit(f"Missing weights: {checkpoint_dir / 'best_model.weights.h5'}")

    # TF perf knobs
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    cfg = _load_config(checkpoint_dir)
    img_size = int(cfg.get("img_size", 224))
    num_bins = int(cfg.get("num_bins", img_size))

    model = load_inference_model(
        checkpoint_dir / "best_model.weights.h5",
        backbone=cfg.get("backbone", "mobilenetv2"),
        alpha=cfg.get("alpha", 0.35),
        fpn_ch=cfg.get("fpn_ch", 32),
        simcc_ch=cfg.get("simcc_ch", 96),
        img_size=img_size,
        num_bins=num_bins,
    )

    print("\n" + "=" * 80)
    print("Generating checkpoint pseudo-label splits")
    print("=" * 80)
    print(f"Dataset: {data_root}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Splits: {splits}")
    print(f"img_size={img_size}  batch_size={args.batch_size}  score_threshold={args.score_threshold}")

    for split in splits:
        split_file = data_root / f"{split}.txt"
        if not split_file.exists():
            raise SystemExit(f"Missing split file: {split_file}")

        out_path = data_root / f"{split}_{args.suffix}.txt"
        if out_path.exists() and not bool(args.overwrite):
            raise SystemExit(f"Refusing to overwrite existing {out_path}. Use --overwrite.")

        names = _read_split_lines(split_file)
        if args.max_images and args.max_images > 0:
            names = names[: int(args.max_images)]

        print(f"\n{split}: {len(names)} entries -> {out_path.name}")

        batch_x: list[np.ndarray] = []
        batch_names: list[str] = []
        batch_is_neg: list[bool] = []

        def flush(fh) -> None:
            if not batch_names:
                return
            x = np.stack(batch_x, axis=0)
            preds = model.predict(x, verbose=0)

            # Inference model outputs [coords, score_logit]
            if isinstance(preds, (list, tuple)) and len(preds) == 2:
                coords_pred = np.asarray(preds[0], dtype=np.float32)  # [B,8]
                score_logit = np.asarray(preds[1], dtype=np.float32)  # [B,1] or [B]
                if score_logit.ndim == 2 and score_logit.shape[1] == 1:
                    score_logit = score_logit[:, 0]
                score_prob = 1.0 / (1.0 + np.exp(-score_logit))
            else:
                # Back-compat: some models may return concatenated [coords, score]
                preds_arr = np.asarray(preds, dtype=np.float32)
                if preds_arr.ndim != 2 or preds_arr.shape[1] < 9:
                    raise RuntimeError(f"Unexpected model output shape: {preds_arr.shape}")
                coords_pred = preds_arr[:, :8].astype(np.float32)
                score_prob = preds_arr[:, 8].astype(np.float32)

            for i, name in enumerate(batch_names):
                is_neg = batch_is_neg[i]
                if is_neg:
                    detected = 0
                    score = 0.0
                    coords = np.full((8,), -1.0, dtype=np.float32)
                else:
                    coords = coords_pred[i].astype(np.float32)
                    score = float(score_prob[i])
                    coords = np.clip(coords, 0.0, 1.0)
                    detected = 1 if score >= float(args.score_threshold) else 0
                    if detected != 1:
                        coords = np.full((8,), -1.0, dtype=np.float32)
                    else:
                        coords = _order_points_tl_tr_br_bl(coords)

                parts = [name, str(int(detected)), f"{float(score):.6f}"] + [f"{float(v):.6f}" for v in coords.tolist()]
                fh.write(" ".join(parts) + "\n")

            batch_x.clear()
            batch_names.clear()
            batch_is_neg.clear()

        with out_path.open("w") as f:
            for name in tqdm(names, desc=f"Infer ({split})"):
                is_neg = name.startswith("negative_")
                if is_neg:
                    # Keep negatives in output (fast path)
                    parts = [name, "0", "0.000000"] + ["-1.000000"] * 8
                    f.write(" ".join(parts) + "\n")
                    continue

                img_dir = data_root / "images"
                img_path = img_dir / name
                if not img_path.exists():
                    # If missing, write as undetected
                    parts = [name, "0", "0.000000"] + ["-1.000000"] * 8
                    f.write(" ".join(parts) + "\n")
                    continue

                try:
                    with Image.open(img_path) as img:
                        x = _preprocess_image(img, img_size)
                except Exception:
                    parts = [name, "0", "0.000000"] + ["-1.000000"] * 8
                    f.write(" ".join(parts) + "\n")
                    continue

                batch_x.append(x)
                batch_names.append(name)
                batch_is_neg.append(False)

                if len(batch_names) >= int(args.batch_size):
                    flush(f)

            flush(f)

    print("\nDone.")


if __name__ == "__main__":
    main()
