#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path


def _read_split(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _write_split(path: Path, names: list[str]) -> None:
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def _parse_prefixes(prefixes_csv: str) -> list[str]:
    return [p.strip().lower() for p in str(prefixes_csv).split(",") if p.strip()]


def _label_has_any_annotation(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove samples whose label file has no annotations (empty/missing) for specific prefixes."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(
            "/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-last"
        ),
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        default="midv,smartdoc",
        help="Comma-separated prefixes (case-insensitive). Default: midv,smartdoc",
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
        help="Where to move removed samples. Defaults to <data_root>/removed_empty_annotations_<timestamp>/",
    )
    args = parser.parse_args()

    data_root: Path = args.data_root
    prefixes = _parse_prefixes(args.prefixes)
    if not prefixes:
        raise SystemExit("No prefixes provided.")

    required = [
        data_root / "images",
        data_root / "labels",
        data_root / "train.txt",
        data_root / "val.txt",
    ]
    for p in required:
        if not p.exists():
            raise SystemExit(f"Missing: {p}")

    train_names = _read_split(data_root / "train.txt")
    val_names = _read_split(data_root / "val.txt")
    train_set = set(train_names)
    val_set = set(val_names)

    labels_dir = data_root / "labels"
    images_dir = data_root / "images"

    def want(name: str) -> bool:
        lower = name.lower()
        return any(lower.startswith(p) for p in prefixes)

    to_remove_train: list[str] = []
    to_remove_val: list[str] = []
    stats = Counter()

    for name in train_names:
        if name.lower().startswith("negative_"):
            continue
        if not want(name):
            continue
        lbl = labels_dir / f"{Path(name).stem}.txt"
        if not _label_has_any_annotation(lbl):
            to_remove_train.append(name)
            stats[f"train::{name.split('_', 1)[0].lower()}"] += 1

    for name in val_names:
        if name.lower().startswith("negative_"):
            continue
        if not want(name):
            continue
        lbl = labels_dir / f"{Path(name).stem}.txt"
        if not _label_has_any_annotation(lbl):
            to_remove_val.append(name)
            stats[f"val::{name.split('_', 1)[0].lower()}"] += 1

    to_remove_all = sorted(set(to_remove_train) | set(to_remove_val))
    if not to_remove_all:
        print("No matching samples with empty/missing annotations found.")
        return

    print(f"Prefixes: {prefixes}")
    print(f"Will remove: {len(to_remove_all)} files")
    print(f"  - train: {len(to_remove_train)}")
    print(f"  - val:   {len(to_remove_val)}")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to execute.")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    removed_dir = (
        args.removed_dir
        if args.removed_dir is not None
        else (data_root / f"removed_empty_annotations_{ts}")
    )
    removed_dir.mkdir(parents=True, exist_ok=True)
    (removed_dir / "images").mkdir(exist_ok=True)
    (removed_dir / "labels").mkdir(exist_ok=True)

    # Backup split files before rewriting.
    shutil.copy2(data_root / "train.txt", removed_dir / "train.txt.bak")
    shutil.copy2(data_root / "val.txt", removed_dir / "val.txt.bak")

    moved_images = 0
    moved_labels = 0
    missing_images: list[str] = []
    missing_labels: list[str] = []

    for name in to_remove_all:
        src_img = images_dir / name
        src_lbl = labels_dir / f"{Path(name).stem}.txt"
        dst_img = removed_dir / "images" / name
        dst_lbl = removed_dir / "labels" / f"{Path(name).stem}.txt"

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
        "prefixes": prefixes,
        "removed_dir": str(removed_dir),
        "removed_filenames": to_remove_all,
        "removed_train": sorted(set(to_remove_train)),
        "removed_val": sorted(set(to_remove_val)),
        "moved_images": moved_images,
        "moved_labels": moved_labels,
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "train_before": len(train_names),
        "train_after": len(train_new),
        "val_before": len(val_names),
        "val_after": len(val_new),
        "counts_by_split_prefix": dict(stats),
    }
    (removed_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (removed_dir / "removed_filenames.txt").write_text(
        "\n".join(to_remove_all) + "\n", encoding="utf-8"
    )

    print(f"\nApplied. Moved images={moved_images}, labels={moved_labels}")
    print(f"Updated splits: train {len(train_names)} -> {len(train_new)} | val {len(val_names)} -> {len(val_new)}")
    print(f"Report: {removed_dir / 'report.json'}")


if __name__ == "__main__":
    main()

