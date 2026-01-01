#!/usr/bin/env python3
"""
Find expansion candidates by comparing old/new evaluation CSVs.

Candidates are samples with old_iou < iou_old_max AND new_iou >= iou_new_min.
Outputs a combined list and per-split lists (e.g., *_train.txt, *_val.txt).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


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


def _load_scores(path: str) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = (row.get("split") or "").strip()
            name = _get_row_name(row)
            if not split or not name:
                continue
            iou = _parse_iou(row.get("iou"))
            if iou is None:
                continue
            scores[(split, name)] = iou
    return scores


def _write_list(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for name in names:
            f.write(name + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find expansion candidates by comparing two evaluation CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv_old", type=str, required=True, help="Baseline evaluation CSV")
    parser.add_argument("--csv_new", type=str, required=True, help="New evaluation CSV")
    parser.add_argument("--iou_old_max", type=float, default=0.99, help="Upper bound for old IoU")
    parser.add_argument("--iou_new_min", type=float, default=0.98, help="Lower bound for new IoU")
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="Comma-separated splits to include",
    )
    parser.add_argument(
        "--negative_prefix",
        type=str,
        default="negative_",
        help="Prefix used to identify negative samples",
    )
    parser.add_argument(
        "--include_negatives",
        action="store_true",
        help="Include negatives in candidates (default: skip negatives)",
    )
    parser.add_argument("--output", type=str, required=True, help="Output candidates list (combined)")
    args = parser.parse_args()

    splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    old_scores = _load_scores(args.csv_old)
    new_scores = _load_scores(args.csv_new)

    combined: list[str] = []
    per_split: dict[str, list[str]] = {split: [] for split in splits}

    for key, old_iou in old_scores.items():
        split, name = key
        if split not in splits:
            continue
        if not args.include_negatives and name.startswith(args.negative_prefix):
            continue
        new_iou = new_scores.get(key)
        if new_iou is None:
            continue
        if old_iou < args.iou_old_max and new_iou >= args.iou_new_min:
            combined.append(name)
            per_split[split].append(name)

    output_path = Path(args.output)
    _write_list(output_path, combined)

    for split, names in per_split.items():
        split_path = output_path.with_name(f"{output_path.stem}_{split}{output_path.suffix}")
        _write_list(split_path, names)

    print(f"Combined: {output_path} ({len(combined)} samples)")
    for split, names in per_split.items():
        split_path = output_path.with_name(f"{output_path.stem}_{split}{output_path.suffix}")
        print(f"{split}: {split_path} ({len(names)} samples)")


if __name__ == "__main__":
    main()
