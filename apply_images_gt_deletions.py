#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _Rec:
    split: str
    name: str
    overlay_rel: str
    is_negative: bool


def _load_index(index_path: Path) -> list[_Rec]:
    if not index_path.exists():
        raise SystemExit(f"Missing index file: {index_path}")
    recs: list[_Rec] = []
    for raw in index_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if "error" in obj:
            continue
        split = str(obj.get("split", "")).strip()
        name = str(obj.get("name", "")).strip()
        overlay_rel = str(obj.get("overlay_rel", "")).strip()
        is_negative = bool(obj.get("is_negative", False))
        if split not in {"train", "val"}:
            continue
        if not name or not overlay_rel:
            continue
        recs.append(_Rec(split=split, name=name, overlay_rel=overlay_rel, is_negative=is_negative))
    return recs


def _read_split(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _write_split(path: Path, names: list[str]) -> None:
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply manual review deletions: overlays removed from images-gt => remove original image/label and update train/val."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(
            "/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-last"
        ),
    )
    parser.add_argument(
        "--review_dir",
        type=Path,
        default=None,
        help="Defaults to <data_root>/images-gt.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Defaults to <review_dir>/index.jsonl (from export_images_gt_overlays.py).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files + rewrite splits. Without this flag, only prints what would happen.",
    )
    parser.add_argument(
        "--removed_dir",
        type=Path,
        default=None,
        help="Where to move removed samples. Defaults to <data_root>/removed_by_review_<timestamp>/",
    )
    args = parser.parse_args()

    data_root: Path = args.data_root
    review_dir = args.review_dir if args.review_dir is not None else (data_root / "images-gt")
    index_path = args.index if args.index is not None else (review_dir / "index.jsonl")

    required = [
        data_root / "images",
        data_root / "labels",
        data_root / "images-negative",
        data_root / "labels-negative",
        data_root / "train.txt",
        data_root / "val.txt",
        review_dir,
        index_path,
    ]
    for p in required:
        if not p.exists():
            raise SystemExit(f"Missing: {p}")

    recs = _load_index(index_path)
    if not recs:
        raise SystemExit(f"No usable records in index: {index_path}")

    train_names = _read_split(data_root / "train.txt")
    val_names = _read_split(data_root / "val.txt")
    train_set = set(train_names)
    val_set = set(val_names)

    deleted_recs: list[_Rec] = []
    for rec in recs:
        overlay_path = review_dir / rec.overlay_rel
        if not overlay_path.exists():
            deleted_recs.append(rec)

    if not deleted_recs:
        print("No deleted overlays detected. Nothing to do.")
        return

    # Only remove samples that are still present in the current splits (idempotent).
    to_remove_train = {r.name for r in deleted_recs if r.split == "train" and r.name in train_set}
    to_remove_val = {r.name for r in deleted_recs if r.split == "val" and r.name in val_set}
    to_remove_all = to_remove_train | to_remove_val
    already_absent = sorted({r.name for r in deleted_recs} - to_remove_all)

    print(f"Deleted overlays detected: {len(deleted_recs)} records")
    print(f"Unique filenames to remove (still in splits): {len(to_remove_all)}")
    print(f"  - train: {len(to_remove_train)}")
    print(f"  - val:   {len(to_remove_val)}")
    if already_absent:
        print(f"Already absent from splits (ignored): {len(already_absent)}")

    if not to_remove_all:
        print("\nNothing new to remove.")
        return

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to move files and rewrite splits.")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    removed_dir = (
        args.removed_dir
        if args.removed_dir is not None
        else (data_root / f"removed_by_review_{ts}")
    )
    removed_dir.mkdir(parents=True, exist_ok=True)
    (removed_dir / "images").mkdir(exist_ok=True)
    (removed_dir / "labels").mkdir(exist_ok=True)
    (removed_dir / "images-negative").mkdir(exist_ok=True)
    (removed_dir / "labels-negative").mkdir(exist_ok=True)

    # Backup split files before rewriting.
    shutil.copy2(data_root / "train.txt", removed_dir / "train.txt.bak")
    shutil.copy2(data_root / "val.txt", removed_dir / "val.txt.bak")

    moved_images = 0
    moved_labels = 0
    missing_images: list[str] = []
    missing_labels: list[str] = []

    # Move original files
    for name in sorted(to_remove_all):
        is_negative = name.lower().startswith("negative_")
        src_img = (data_root / "images-negative" / name) if is_negative else (data_root / "images" / name)
        src_lbl = (
            (data_root / "labels-negative" / f"{Path(name).stem}.txt")
            if is_negative
            else (data_root / "labels" / f"{Path(name).stem}.txt")
        )
        dst_img = (
            (removed_dir / "images-negative" / name)
            if is_negative
            else (removed_dir / "images" / name)
        )
        dst_lbl = (
            (removed_dir / "labels-negative" / f"{Path(name).stem}.txt")
            if is_negative
            else (removed_dir / "labels" / f"{Path(name).stem}.txt")
        )

        if src_img.exists():
            src_img.rename(dst_img)
            moved_images += 1
        else:
            missing_images.append(name)

        if src_lbl.exists():
            src_lbl.rename(dst_lbl)
            moved_labels += 1
        else:
            missing_labels.append(f"{Path(name).stem}.txt")

    # Rewrite splits
    train_new = [n for n in train_names if n not in to_remove_all]
    val_new = [n for n in val_names if n not in to_remove_all]
    _write_split(data_root / "train.txt", train_new)
    _write_split(data_root / "val.txt", val_new)

    report = {
        "data_root": str(data_root),
        "review_dir": str(review_dir),
        "index": str(index_path),
        "removed_dir": str(removed_dir),
        "removed_filenames": sorted(to_remove_all),
        "removed_train": sorted(to_remove_train),
        "removed_val": sorted(to_remove_val),
        "already_absent_from_splits": already_absent,
        "moved_images": moved_images,
        "moved_labels": moved_labels,
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "train_before": len(train_names),
        "train_after": len(train_new),
        "val_before": len(val_names),
        "val_after": len(val_new),
    }
    (removed_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nApplied. Moved images={moved_images}, labels={moved_labels}")
    print(f"Updated splits: train {len(train_names)} -> {len(train_new)} | val {len(val_names)} -> {len(val_new)}")
    print(f"Report: {removed_dir / 'report.json'}")


if __name__ == "__main__":
    main()
