#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _read_split(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _load_iou_csv(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            fn = row.get("filename")
            if not fn:
                continue
            try:
                out[fn] = float(row["iou"])
            except Exception:
                continue
    return out


def _load_teacher_preds(path: Path) -> dict[str, dict[str, Any]]:
    """
    Parse lines like:
      filename detected conf x0 y0 x1 y1 x2 y2 x3 y3
    """
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 3:
                continue
            name = parts[0]
            try:
                detected = int(float(parts[1]))
            except Exception:
                detected = 0
            try:
                conf = float(parts[2])
            except Exception:
                conf = 0.0
            coords: list[float] | None = None
            if detected == 1 and len(parts) >= 11:
                try:
                    coords = [float(x) for x in parts[3:11]]
                except Exception:
                    coords = None
            out[name] = {"detected": detected, "conf": conf, "coords": coords}
    return out


def _is_protected_prefix(name: str, prefixes: list[str]) -> bool:
    low = name.lower()
    return any(low.startswith(p) for p in prefixes)


def _write_label(path: Path, coords_norm8: list[float] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not coords_norm8:
        path.write_text("", encoding="utf-8")
        return
    parts = ["0"] + [f"{float(v):.6f}" for v in coords_norm8]
    path.write_text(" ".join(parts) + "\n", encoding="utf-8")


def _copy_one(src: Path, dst: Path, overwrite: bool) -> str | None:
    if dst.exists() and not overwrite:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_root",
        type=Path,
        default=Path(
            "/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new"
        ),
    )
    parser.add_argument(
        "--dst_root",
        type=Path,
        default=Path(
            "/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-last"
        ),
    )
    parser.add_argument("--iou_threshold", type=float, default=0.97)
    parser.add_argument(
        "--protected_prefixes",
        type=str,
        default="midv,smartdoc",
        help="Comma-separated lowercase prefixes to always include.",
    )
    parser.add_argument("--train_split", type=str, default="train_final_pruned")
    parser.add_argument("--val_split", type=str, default="val_final_pruned")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    src_root: Path = args.src_root
    dst_root: Path = args.dst_root

    prefixes = [p.strip().lower() for p in args.protected_prefixes.split(",") if p.strip()]

    src_images = src_root / "images"
    src_labels = src_root / "labels"
    src_neg_images = src_root / "images-negative"
    src_neg_labels = src_root / "labels-negative"

    train_split_path = src_root / f"{args.train_split}.txt"
    val_split_path = src_root / f"{args.val_split}.txt"

    train_iou_path = src_root / "train_gt_vs_teacher_iou.csv"
    val_iou_path = src_root / "val_gt_vs_teacher_iou.csv"
    train_pred_path = src_root / "train_teacher.txt"
    val_pred_path = src_root / "val_teacher.txt"

    for p in [
        src_images,
        src_labels,
        src_neg_images,
        src_neg_labels,
        train_split_path,
        val_split_path,
        train_iou_path,
        val_iou_path,
        train_pred_path,
        val_pred_path,
    ]:
        if not p.exists():
            raise SystemExit(f"Missing: {p}")

    if dst_root.exists():
        non_empty = any(dst_root.iterdir())
        if non_empty:
            raise SystemExit(
                f"Destination not empty: {dst_root} (move it away or empty it first)"
            )
    dst_root.mkdir(parents=True, exist_ok=True)

    dst_images = dst_root / "images"
    dst_labels = dst_root / "labels"
    dst_neg_images = dst_root / "images-negative"
    dst_neg_labels = dst_root / "labels-negative"

    for d in [dst_images, dst_labels, dst_neg_images, dst_neg_labels]:
        d.mkdir(parents=True, exist_ok=True)

    train_names = _read_split(train_split_path)
    val_names = _read_split(val_split_path)

    iou_train = _load_iou_csv(train_iou_path)
    iou_val = _load_iou_csv(val_iou_path)

    pred_train = _load_teacher_preds(train_pred_path)
    pred_val = _load_teacher_preds(val_pred_path)

    def should_include(name: str, split: str) -> bool:
        low = name.lower()
        if _is_protected_prefix(low, prefixes):
            return True
        if low.startswith("negative_"):
            return True
        iou_map = iou_train if split == "train" else iou_val
        iou = iou_map.get(name)
        return iou is not None and iou >= args.iou_threshold

    train_kept = [n for n in train_names if should_include(n, "train")]
    val_kept = [n for n in val_names if should_include(n, "val")]

    # Copy images
    copy_tasks: list[tuple[Path, Path]] = []
    for split, names in [("train", train_kept), ("val", val_kept)]:
        for name in names:
            low = name.lower()
            if low.startswith("negative_"):
                src = src_neg_images / name
                dst = dst_neg_images / name
            else:
                src = src_images / name
                dst = dst_images / name
            if not src.exists():
                # Fallback (rare): try the other folder
                alt = (src_images / name) if src.parent == src_neg_images else (src_neg_images / name)
                if alt.exists():
                    src = alt
                else:
                    continue
            copy_tasks.append((src, dst))

    copied = 0
    skipped = 0
    missing_src = 0

    pbar = tqdm(total=len(copy_tasks), desc="Copying images") if tqdm else None
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = [ex.submit(_copy_one, src, dst, args.overwrite) for (src, dst) in copy_tasks]
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception:
                missing_src += 1
                if pbar:
                    pbar.update(1)
                continue
            if res is None:
                skipped += 1
            else:
                copied += 1
            if pbar:
                pbar.update(1)
    if pbar:
        pbar.close()

    # Write labels from teacher inference
    label_written = 0
    label_empty = 0
    label_missing_pred = 0

    def write_for(split: str, name: str) -> None:
        nonlocal label_written, label_empty, label_missing_pred
        low = name.lower()
        if low.startswith("negative_"):
            # Keep negatives as negatives (empty labels).
            _write_label(dst_neg_labels / f"{Path(name).stem}.txt", None)
            label_empty += 1
            return

        pred_map = pred_train if split == "train" else pred_val
        pred = pred_map.get(name)
        if not pred:
            label_missing_pred += 1
            _write_label(dst_labels / f"{Path(name).stem}.txt", None)
            return

        coords = pred.get("coords") if int(pred.get("detected", 0)) == 1 else None
        if coords:
            _write_label(dst_labels / f"{Path(name).stem}.txt", coords)
            label_written += 1
        else:
            _write_label(dst_labels / f"{Path(name).stem}.txt", None)
            label_empty += 1

    all_label_jobs: list[tuple[str, str]] = [("train", n) for n in train_kept] + [
        ("val", n) for n in val_kept
    ]
    pbar2 = tqdm(total=len(all_label_jobs), desc="Writing labels") if tqdm else None
    for split, name in all_label_jobs:
        write_for(split, name)
        if pbar2:
            pbar2.update(1)
    if pbar2:
        pbar2.close()

    # Write split files
    (dst_root / "train.txt").write_text("\n".join(train_kept) + "\n", encoding="utf-8")
    (dst_root / "val.txt").write_text("\n".join(val_kept) + "\n", encoding="utf-8")

    # Summary
    def breakdown(names: list[str]) -> dict[str, int]:
        from collections import Counter

        c = Counter()
        for n in names:
            low = n.lower()
            if low.startswith("negative_"):
                c["negative"] += 1
            elif low.startswith("midv"):
                c["midv"] += 1
            elif low.startswith("smartdoc"):
                c["smartdoc"] += 1
            else:
                c["other_iou_ge"] += 1
        return dict(c)

    summary = {
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "iou_threshold": float(args.iou_threshold),
        "protected_prefixes": prefixes,
        "splits": {
            "train_src": str(train_split_path),
            "val_src": str(val_split_path),
            "train_count_src": len(train_names),
            "val_count_src": len(val_names),
            "train_count_kept": len(train_kept),
            "val_count_kept": len(val_kept),
            "train_breakdown": breakdown(train_kept),
            "val_breakdown": breakdown(val_kept),
        },
        "copy": {
            "tasks": len(copy_tasks),
            "copied": copied,
            "skipped": skipped,
            "missing_src": missing_src,
        },
        "labels": {
            "written": label_written,
            "empty": label_empty,
            "missing_pred": label_missing_pred,
        },
    }
    (dst_root / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done.")
    print("Train kept:", len(train_kept))
    print("Val kept:", len(val_kept))
    print("Summary:", dst_root / "build_summary.json")


if __name__ == "__main__":
    main()
