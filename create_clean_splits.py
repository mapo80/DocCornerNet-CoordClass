#!/usr/bin/env python3
"""
Create clean train/val split files from an evaluation CSV.

The CSV is expected to contain at least: split, filename, iou.
Negative samples are re-added by sampling from the original split files
to preserve the negative/positive ratio (unless overridden).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path


def _find_split_file(data_root: Path, split: str) -> Path:
    for suffix in ("_with_negative_v2", "_with_negative", ""):
        candidate = data_root / f"{split}{suffix}.txt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No split file found for '{split}' in {data_root}")


def _load_split_names(data_root: Path, split: str) -> list[str]:
    path = _find_split_file(data_root, split)
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _get_row_name(row: dict[str, str]) -> str | None:
    for key in ("filename", "image", "name", "file"):
        value = row.get(key)
        if value:
            return value.strip()
    return None


def _parse_iou(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        iou = float(value)
    except ValueError:
        return None
    if not math.isfinite(iou):
        return None
    return iou


def _negative_ratio(names: list[str], negative_prefix: str) -> float:
    neg = sum(1 for name in names if name.startswith(negative_prefix))
    pos = max(1, len(names) - neg)
    return neg / pos


def _sample_negatives(
    split_names: list[str],
    positive_count: int,
    negative_prefix: str,
    rng: random.Random,
    neg_ratio: float,
) -> list[str]:
    negatives = [name for name in split_names if name.startswith(negative_prefix)]
    target_count = int(round(positive_count * neg_ratio))
    if target_count <= 0 or not negatives:
        return []
    target_count = min(target_count, len(negatives))
    return rng.sample(negatives, target_count)


def _write_split(output_path: Path, names: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for name in names:
            f.write(name + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create clean train/val splits from full_evaluation.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to full_evaluation.csv")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root (for original splits)")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to write split files")
    parser.add_argument("--iou_threshold", type=float, default=0.99, help="IoU threshold for clean positives")
    parser.add_argument("--train_split", type=str, default="train", help="Train split name")
    parser.add_argument("--val_split", type=str, default="val", help="Val split name")
    parser.add_argument("--train_out", type=str, default="train_clean", help="Train output split name")
    parser.add_argument("--val_out", type=str, default="val_clean", help="Val output split name")
    parser.add_argument(
        "--train_neg_ratio",
        type=float,
        default=None,
        help="Override negative/positive ratio for train (defaults to original split ratio)",
    )
    parser.add_argument(
        "--val_neg_ratio",
        type=float,
        default=None,
        help="Override negative/positive ratio for val (defaults to original split ratio)",
    )
    parser.add_argument(
        "--negative_prefix",
        type=str,
        default="negative_",
        help="Prefix used to identify negative samples",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for negative sampling")
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Do not shuffle output split lists",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else data_root
    rng = random.Random(args.seed)

    train_pos: list[str] = []
    val_pos: list[str] = []

    with open(args.csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = (row.get("split") or "").strip()
            name = _get_row_name(row)
            if not split or not name:
                continue
            if name.startswith(args.negative_prefix):
                continue
            iou = _parse_iou(row.get("iou"))
            if iou is None or iou < args.iou_threshold:
                continue
            if split == args.train_split:
                train_pos.append(name)
            elif split == args.val_split:
                val_pos.append(name)

    train_names = _load_split_names(data_root, args.train_split)
    val_names = _load_split_names(data_root, args.val_split)

    train_ratio = args.train_neg_ratio
    if train_ratio is None:
        train_ratio = _negative_ratio(train_names, args.negative_prefix)
    val_ratio = args.val_neg_ratio
    if val_ratio is None:
        val_ratio = _negative_ratio(val_names, args.negative_prefix)

    train_neg = _sample_negatives(
        train_names, len(train_pos), args.negative_prefix, rng, train_ratio
    )
    val_neg = _sample_negatives(
        val_names, len(val_pos), args.negative_prefix, rng, val_ratio
    )

    train_out = train_pos + train_neg
    val_out = val_pos + val_neg
    if not args.no_shuffle:
        rng.shuffle(train_out)
        rng.shuffle(val_out)

    train_path = output_dir / f"{args.train_out}.txt"
    val_path = output_dir / f"{args.val_out}.txt"
    _write_split(train_path, train_out)
    _write_split(val_path, val_out)

    print(f"CSV: {args.csv}")
    print(f"Train positives: {len(train_pos)}")
    print(f"Train negatives: {len(train_neg)} (ratio={train_ratio:.3f})")
    print(f"Train output: {train_path} ({len(train_out)} total)")
    print(f"Val positives: {len(val_pos)}")
    print(f"Val negatives: {len(val_neg)} (ratio={val_ratio:.3f})")
    print(f"Val output: {val_path} ({len(val_out)} total)")


if __name__ == "__main__":
    main()
