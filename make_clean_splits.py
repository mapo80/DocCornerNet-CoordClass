"""
Create cleaned train/val split files to reduce label noise / hard outliers.

This script generates:
  - <train_out>_with_negative_v2.txt and <train_out>.txt
  - <val_out>_with_negative_v2.txt and <val_out>.txt

Cleaning strategy:
  1) Label sanity filter (always): remove positives with invalid/degenerate quads.
  2) Val target (optional, default on): using a scoring model, drop the *minimum*
     number of lowest-IoU positives until mean IoU >= target.
  3) Train cleaning (optional): drop a fraction of highest corner-error positives
     (fast heuristic) + optionally drop names from outliers.txt.

Notes:
  - IoU/corner error are computed only on positives (has_doc == 1).
  - Negatives are kept by default.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf

from dataset import create_dataset, load_split_file
from evaluate import load_model
from metrics import compute_corner_error, compute_polygon_iou


def parse_args():
    p = argparse.ArgumentParser(
        description="Create cleaned train/val split files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_root", type=str, required=True, help="Dataset root")

    # Base split names (existing files are resolved with *_with_negative_v2 fallback)
    p.add_argument("--train_in", type=str, default="train", help="Base train split name")
    p.add_argument("--val_in", type=str, default="val", help="Base val split name")

    # Output split names (files are written to data_root)
    p.add_argument("--train_out", type=str, default="train_cleaned", help="Output train split name")
    p.add_argument("--val_out", type=str, default="val_cleaned", help="Output val split name")

    # Model used to score samples (required for val target; optional for label-only cleaning)
    p.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Checkpoint dir/.keras or .weights.h5 used to score IoU/errors for cleaning.",
    )
    p.add_argument(
        "--input_norm",
        type=str,
        default="imagenet",
        choices=["imagenet", "zero_one", "raw255"],
        help="Input normalization for the scoring model (must match how it was trained).",
    )
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for scoring model forward pass")

    # Cleaning controls
    p.add_argument("--keep_negatives", action="store_true", help="Keep all negative samples (recommended)")
    p.set_defaults(keep_negatives=True)
    p.add_argument(
        "--drop_outliers_txt",
        action="store_true",
        help="Also drop positives listed in data_root/outliers.txt (if present) from both splits.",
    )
    p.set_defaults(drop_outliers_txt=True)

    # Val: target mean IoU
    p.add_argument("--target_mean_iou", type=float, default=0.99, help="Target mean IoU on val positives")
    p.add_argument(
        "--max_val_drop_frac",
        type=float,
        default=0.08,
        help="Safety cap: max fraction of val positives that can be dropped to reach the target",
    )

    # Train: heuristic cleaning (corner error)
    p.add_argument(
        "--train_drop_frac",
        type=float,
        default=None,
        help="Fraction of train positives to drop (highest corner error). If None, uses val drop fraction.",
    )
    p.add_argument(
        "--train_corner_err_px_threshold",
        type=float,
        default=None,
        help="If set, additionally drop train positives with corner error > threshold (px).",
    )

    # Output report
    p.add_argument("--report_dir", type=str, default=None, help="Where to write JSON reports (default: data_root/.cleaning)")
    return p.parse_args()


def _find_split_file(data_root: Path, split: str) -> Path:
    for prefix in (f"{split}_with_negative_v2", f"{split}_with_negative", split):
        candidate = data_root / f"{prefix}.txt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No split file found for split='{split}' in {data_root}")


def _load_outliers_txt(data_root: Path) -> set[str]:
    path = data_root / "outliers.txt"
    if not path.exists():
        return set()
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def _quad_is_sane(coords: np.ndarray) -> bool:
    """
    Very cheap label sanity checks (coords are normalized [0,1]).
    Reject clearly broken/degenerate labels.
    """
    if coords.shape != (8,):
        return False
    if not np.isfinite(coords).all():
        return False
    if (coords < -0.05).any() or (coords > 1.05).any():
        return False
    pts = coords.reshape(4, 2)
    # Area check via shoelace on the given order.
    x = pts[:, 0]
    y = pts[:, 1]
    area = 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area < 1e-4:  # extremely tiny in normalized space
        return False
    # Bounding box size sanity
    w = float(x.max() - x.min())
    h = float(y.max() - y.min())
    if w < 1e-3 or h < 1e-3:
        return False
    return True


@dataclass
class SplitScore:
    names: list[str]
    has_doc: np.ndarray  # [N]
    coords_gt: np.ndarray  # [N,8]
    coords_pred: np.ndarray  # [N,8]
    scores: np.ndarray  # [N]


def _score_split(
    model: tf.keras.Model,
    data_root: str,
    split_name: str,
    input_norm: str,
    batch_size: int,
) -> SplitScore:
    data_root_p = Path(data_root)
    split_file = _find_split_file(data_root_p, split_name)
    names = load_split_file(str(split_file))

    ds = create_dataset(
        data_root=data_root,
        split=split_name,
        img_size=model.input_shape[1],
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        drop_remainder=False,
        image_norm=input_norm,
    )

    coords_gt_all = []
    has_doc_all = []
    coords_pred_all = []
    score_all = []

    n_seen = 0
    for images, targets in ds:
        outputs = model(images, training=False)
        if isinstance(outputs, dict):
            coords_pred = outputs["coords"].numpy()
            score_logit = outputs["score_logit"].numpy()
        else:
            coords_pred = outputs[0].numpy()
            score_logit = outputs[1].numpy()

        score = 1.0 / (1.0 + np.exp(-np.clip(score_logit.squeeze(-1), -60.0, 60.0)))
        coords_gt = targets["coords"].numpy()
        has_doc = targets["has_doc"].numpy().astype(np.float32)
        if has_doc.ndim == 2:
            has_doc = has_doc.squeeze(-1)

        b = int(coords_gt.shape[0])
        coords_gt_all.append(coords_gt)
        has_doc_all.append(has_doc)
        coords_pred_all.append(coords_pred)
        score_all.append(score)

        n_seen += b
        if n_seen >= len(names):
            break

    return SplitScore(
        names=names,
        has_doc=np.concatenate(has_doc_all, axis=0)[: len(names)],
        coords_gt=np.concatenate(coords_gt_all, axis=0)[: len(names)],
        coords_pred=np.concatenate(coords_pred_all, axis=0)[: len(names)],
        scores=np.concatenate(score_all, axis=0)[: len(names)],
    )


def _write_split_files(data_root: Path, split_out: str, names: Iterable[str]):
    names = list(names)
    out_main = data_root / f"{split_out}_with_negative_v2.txt"
    out_alt = data_root / f"{split_out}.txt"
    for p in (out_main, out_alt):
        with open(p, "w") as f:
            for n in names:
                f.write(f"{n}\n")
    return out_main, out_alt


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    report_dir = Path(args.report_dir) if args.report_dir else (data_root / ".cleaning")
    report_dir.mkdir(parents=True, exist_ok=True)

    outliers_txt = _load_outliers_txt(data_root) if args.drop_outliers_txt else set()
    if outliers_txt:
        print(f"Loaded outliers.txt: {len(outliers_txt)} names (will be dropped from positives)")

    # Load input split files
    train_file = _find_split_file(data_root, args.train_in)
    val_file = _find_split_file(data_root, args.val_in)
    train_names = load_split_file(str(train_file))
    val_names = load_split_file(str(val_file))
    print(f"Train in: {train_file.name} ({len(train_names)} samples)")
    print(f"Val in:   {val_file.name} ({len(val_names)} samples)")

    if not args.model_path:
        raise SystemExit("--model_path is required to reach a target_mean_iou via model-based cleaning.")

    # Load scoring model
    class _Args:
        # minimal adapter for evaluate.load_model
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    model_args = _Args(
        model_path=args.model_path,
        data_root=str(data_root),
        split="val",
        batch_size=args.batch_size,
        input_norm=args.input_norm,
        backbone="mobilenetv2",
        alpha=0.35,
        backbone_minimalistic=False,
        backbone_include_preprocessing=False,
        backbone_weights=None,
        fpn_ch=32,
        simcc_ch=96,
        img_size=256,
        num_bins=256,
        tau=1.0,
        unsafe_load=False,
    )
    model, img_size, _ = load_model(model_args)
    print(f"Scoring model img_size={img_size}, params={model.count_params():,}")

    # Score val split
    val_score = _score_split(model, str(data_root), args.val_in, args.input_norm, args.batch_size)

    # Build label sanity mask for positives + drop outliers.txt
    val_is_pos = val_score.has_doc >= 0.5
    val_sane = np.zeros_like(val_is_pos, dtype=bool)
    for i, name in enumerate(val_score.names):
        if not val_is_pos[i]:
            continue
        if name in outliers_txt:
            continue
        val_sane[i] = _quad_is_sane(val_score.coords_gt[i])

    # Compute IoU for sane positives
    val_pos_idx = np.where(val_sane)[0]
    val_ious = np.zeros((len(val_pos_idx),), dtype=np.float32)
    val_errs = np.zeros((len(val_pos_idx),), dtype=np.float32)
    for j, i in enumerate(val_pos_idx):
        val_ious[j] = float(compute_polygon_iou(val_score.coords_pred[i], val_score.coords_gt[i]))
        val_errs[j] = float(compute_corner_error(val_score.coords_pred[i], val_score.coords_gt[i], img_size=img_size)[0])

    base_mean = float(val_ious.mean()) if len(val_ious) else 0.0
    print(f"\nVal positives (sane) count: {len(val_ious)}  mean IoU: {base_mean:.4f}")

    # Drop minimum number of lowest IoU positives until reaching target mean IoU
    order = np.argsort(val_ious)  # ascending IoU within val_pos_idx
    total_sum = float(val_ious.sum())
    n = int(len(val_ious))
    removed = []
    removed_sum = 0.0
    removed_count = 0

    def current_mean():
        keep_n = n - removed_count
        return (total_sum - removed_sum) / max(keep_n, 1)

    while n - removed_count > 0 and current_mean() < args.target_mean_iou:
        k = int(order[removed_count])
        removed_sum += float(val_ious[k])
        removed.append(int(val_pos_idx[k]))
        removed_count += 1
        if removed_count / max(n, 1) > args.max_val_drop_frac:
            break

    reached = current_mean() >= args.target_mean_iou
    val_drop_frac = removed_count / max(n, 1)
    print(f"Val target: {args.target_mean_iou:.3f}  reached={reached}  drop={removed_count}/{n} ({val_drop_frac*100:.2f}%)")
    print(f"Val cleaned mean IoU (by construction): {current_mean():.4f}")

    removed_val_names = {val_score.names[i] for i in removed}

    # Build final val names: keep all negatives + keep sane positives minus removed
    val_out = []
    val_removed_report = []
    for i, name in enumerate(val_score.names):
        if val_score.has_doc[i] < 0.5:
            if args.keep_negatives:
                val_out.append(name)
            continue
        # positives
        if name in outliers_txt:
            val_removed_report.append({"name": name, "reason": "outliers.txt"})
            continue
        if not _quad_is_sane(val_score.coords_gt[i]):
            val_removed_report.append({"name": name, "reason": "label_invalid"})
            continue
        if name in removed_val_names:
            val_removed_report.append({"name": name, "reason": "low_iou_to_hit_target"})
            continue
        val_out.append(name)

    # Train cleaning: score split and drop by corner error (fast, no shapely IoU needed)
    train_score = _score_split(model, str(data_root), args.train_in, args.input_norm, args.batch_size)
    train_is_pos = train_score.has_doc >= 0.5

    # Sanity filter + outliers.txt first
    keep_train = np.ones((len(train_score.names),), dtype=bool)
    for i, name in enumerate(train_score.names):
        if train_score.has_doc[i] < 0.5:
            keep_train[i] = bool(args.keep_negatives)
            continue
        if name in outliers_txt:
            keep_train[i] = False
            continue
        if not _quad_is_sane(train_score.coords_gt[i]):
            keep_train[i] = False

    train_pos_idx = np.where(keep_train & train_is_pos)[0]
    # Corner error in px for positives
    train_errs = np.zeros((len(train_pos_idx),), dtype=np.float32)
    for j, i in enumerate(train_pos_idx):
        train_errs[j] = float(compute_corner_error(train_score.coords_pred[i], train_score.coords_gt[i], img_size=img_size)[0])

    # Drop top fraction by error
    drop_frac = float(args.train_drop_frac) if args.train_drop_frac is not None else float(val_drop_frac)
    drop_k = int(math.floor(drop_frac * len(train_pos_idx)))
    if drop_k > 0:
        worst = np.argsort(train_errs)[::-1][:drop_k]
        for k in worst:
            keep_train[int(train_pos_idx[int(k)])] = False

    if args.train_corner_err_px_threshold is not None:
        thr = float(args.train_corner_err_px_threshold)
        for j, i in enumerate(train_pos_idx):
            if keep_train[i] and float(train_errs[j]) > thr:
                keep_train[i] = False

    train_out = [n for i, n in enumerate(train_score.names) if keep_train[i]]

    # Write splits
    train_paths = _write_split_files(data_root, args.train_out, train_out)
    val_paths = _write_split_files(data_root, args.val_out, val_out)

    # Reports
    report = {
        "model_path": args.model_path,
        "input_norm": args.input_norm,
        "img_size": int(img_size),
        "train_in": args.train_in,
        "val_in": args.val_in,
        "train_out": args.train_out,
        "val_out": args.val_out,
        "val_target_mean_iou": float(args.target_mean_iou),
        "val_base_mean_iou_sane": float(base_mean),
        "val_cleaned_mean_iou_sane": float(current_mean()),
        "val_drop_frac_sane": float(val_drop_frac),
        "train_drop_frac_pos": float(drop_frac),
        "outliers_txt_count": int(len(outliers_txt)),
        "val_removed_count": int(len(val_removed_report)),
        "val_out_count": int(len(val_out)),
        "train_out_count": int(len(train_out)),
    }

    with open(report_dir / "clean_splits_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(report_dir / "val_removed.json", "w") as f:
        json.dump(val_removed_report, f, indent=2)

    with open(report_dir / "val_removed.txt", "w") as f:
        for item in val_removed_report:
            f.write(f"{item['name']}\n")

    print("\nWrote:")
    print(f"  Train: {train_paths[0]} and {train_paths[1]}")
    print(f"  Val:   {val_paths[0]} and {val_paths[1]}")
    print(f"  Report: {report_dir / 'clean_splits_report.json'}")


if __name__ == "__main__":
    main()
