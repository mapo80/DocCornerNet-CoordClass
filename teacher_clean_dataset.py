#!/usr/bin/env python3
"""
Teacher-assisted dataset cleaning for DocScanner dataset (YOLO polygon labels).

Goal:
  - Compare GT annotations vs teacher predictions (DocumentDetector-compatible)
  - Remove *clearly wrong* samples from train/val splits (positives + negatives)
  - Optionally fix GT corner ordering to match teacher (TL,TR,BR,BL) when safe

This script is designed to run locally.
It updates split files by default (with backups).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid

    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False


# -----------------------------
# Worker globals / initialization
# -----------------------------

_G_DATA_ROOT: Optional[Path] = None
_G_DETECTOR = None


def _init_worker(teacher_repo: str, backbone_path: str, head_path: str, data_root: str):
    global _G_DATA_ROOT, _G_DETECTOR
    _G_DATA_ROOT = Path(data_root)

    teacher_repo_path = Path(teacher_repo)
    if str(teacher_repo_path) not in sys.path:
        sys.path.insert(0, str(teacher_repo_path))

    from document_detector import DocumentDetector  # type: ignore

    _G_DETECTOR = DocumentDetector(
        backbone_path=backbone_path,
        head_path=head_path,
        verbose=False,
    )


@dataclass(frozen=True)
class Task:
    split: str
    filename: str
    is_negative: bool


def _load_yolo_polygon(label_path: Path) -> Optional[np.ndarray]:
    """
    Returns normalized coords [8] or None if missing/invalid.
    Format: class x0 y0 x1 y1 x2 y2 x3 y3
    """
    if not label_path.exists():
        return None
    try:
        line = label_path.read_text().strip().splitlines()[0].strip()
    except Exception:
        return None
    if not line:
        return None
    parts = line.split()
    if len(parts) < 9:
        return None
    if parts[0] != "0":
        return None
    try:
        coords = np.array([float(x) for x in parts[1:9]], dtype=np.float32)
    except Exception:
        return None
    return coords


def _polygon_iou(points_a: np.ndarray, points_b: np.ndarray) -> float:
    if points_a.shape != (4, 2) or points_b.shape != (4, 2):
        return 0.0
    if not _HAS_SHAPELY:
        # Very rough bbox IoU fallback
        ax1, ay1 = points_a[:, 0].min(), points_a[:, 1].min()
        ax2, ay2 = points_a[:, 0].max(), points_a[:, 1].max()
        bx1, by1 = points_b[:, 0].min(), points_b[:, 1].min()
        bx2, by2 = points_b[:, 0].max(), points_b[:, 1].max()
        x1, y1 = max(ax1, bx1), max(ay1, by1)
        x2, y2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    def to_poly(pts: np.ndarray) -> "Polygon":
        poly = Polygon([(float(x), float(y)) for x, y in pts])
        if not poly.is_valid:
            poly = make_valid(poly)
            if poly.geom_type == "GeometryCollection":
                polys = [g for g in poly.geoms if g.geom_type == "Polygon"]
                if polys:
                    poly = max(polys, key=lambda p: p.area)
                else:
                    poly = Polygon([(float(x), float(y)) for x, y in pts]).convex_hull
            elif poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda p: p.area)
        return poly

    try:
        pa = to_poly(points_a)
        pb = to_poly(points_b)
        if pa.is_empty or pb.is_empty:
            return 0.0
        inter = pa.intersection(pb).area
        union = pa.union(pb).area
        if union <= 0:
            return 0.0
        return float(inter / union)
    except Exception:
        return 0.0


def _corner_errors_best_rotation(pred_xy: np.ndarray, gt_xy: np.ndarray) -> tuple[float, float]:
    """
    pred_xy, gt_xy: [4,2] in pixel coords.
    Returns mean_err, max_err (best rotation).
    """
    if pred_xy.shape != (4, 2) or gt_xy.shape != (4, 2):
        return 1e9, 1e9
    best = None
    best_sum = float("inf")
    for rot in range(4):
        rp = np.roll(pred_xy, rot, axis=0)
        errs = np.linalg.norm(rp - gt_xy, axis=1)
        s = float(errs.sum())
        if s < best_sum:
            best_sum = s
            best = errs
    if best is None:
        return 1e9, 1e9
    return float(best.mean()), float(best.max())


def _best_gt_rotation_to_match_pred(gt_xy: np.ndarray, pred_xy: np.ndarray) -> tuple[int, float]:
    """
    Find rotation of GT (roll left) that best matches pred order (TL,TR,BR,BL).
    Returns (rot, mean_err_px) where rot in [0..3] and gt_rot = roll(gt, -rot).
    """
    if gt_xy.shape != (4, 2) or pred_xy.shape != (4, 2):
        return 0, 1e9
    best_rot = 0
    best_sum = float("inf")
    best_mean = 1e9
    for rot in range(4):
        gt_rot = np.roll(gt_xy, -rot, axis=0)
        errs = np.linalg.norm(gt_rot - pred_xy, axis=1)
        s = float(errs.sum())
        if s < best_sum:
            best_sum = s
            best_rot = rot
            best_mean = float(errs.mean())
    return best_rot, best_mean


def _process_one(task: Task) -> dict:
    if _G_DATA_ROOT is None or _G_DETECTOR is None:
        raise RuntimeError("Worker not initialized")

    data_root = _G_DATA_ROOT
    detector = _G_DETECTOR

    filename = task.filename
    is_negative = task.is_negative

    img_dir = data_root / ("images-negative" if is_negative else "images")
    lbl_dir = data_root / ("labels-negative" if is_negative else "labels")

    img_path = img_dir / filename
    label_path = lbl_dir / (Path(filename).stem + ".txt")

    out: dict = {
        "split": task.split,
        "filename": filename,
        "is_negative": int(is_negative),
        "image_exists": int(img_path.exists()),
        "label_exists": int(label_path.exists()),
        "teacher_status": "",
        "teacher_conf": 0.0,
        "teacher_detected": 0,
        "gt_valid": 0,
        "gt_oob": 0,
        "gt_area_ratio": "",
        "iou": "",
        "err_mean_px": "",
        "err_max_px": "",
        "gt_rot_to_teacher": "",
        "w": "",
        "h": "",
    }

    if not img_path.exists():
        return out

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            out["w"] = int(w)
            out["h"] = int(h)

            gt_norm = _load_yolo_polygon(label_path) if not is_negative else None
            if gt_norm is not None:
                out["gt_valid"] = 1
                # GT sanity
                tol = 0.02
                out["gt_oob"] = int(bool((gt_norm < -tol).any() or (gt_norm > 1.0 + tol).any()))
                gt_xy = gt_norm.reshape(4, 2) * np.array([[w, h]], dtype=np.float32)
                # area ratio
                if _HAS_SHAPELY:
                    try:
                        poly = Polygon([(float(x), float(y)) for x, y in gt_xy])
                        if not poly.is_valid:
                            poly = make_valid(poly)
                            if poly.geom_type == "GeometryCollection":
                                polys = [g for g in poly.geoms if g.geom_type == "Polygon"]
                                if polys:
                                    poly = max(polys, key=lambda p: p.area)
                                else:
                                    poly = Polygon([(float(x), float(y)) for x, y in gt_xy]).convex_hull
                            elif poly.geom_type == "MultiPolygon":
                                poly = max(poly.geoms, key=lambda p: p.area)
                        area = float(poly.area) if not poly.is_empty else 0.0
                    except Exception:
                        area = 0.0
                else:
                    # shoelace on given order (approx)
                    x = gt_xy[:, 0]
                    y = gt_xy[:, 1]
                    area = 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
                out["gt_area_ratio"] = float(area / max(1.0, float(w) * float(h)))
            else:
                gt_xy = None

            # Run teacher
            result = detector.detect(img)
            conf = float(result.get("confidence", 0.0) or 0.0)
            status = str(result.get("status", ""))
            corners = result.get("corners", None)
            out["teacher_status"] = status
            out["teacher_conf"] = conf
            out["teacher_detected"] = int(corners is not None)

            if corners is not None and gt_norm is not None:
                pred_xy = np.array([(float(x), float(y)) for x, y in corners], dtype=np.float32)
                gt_xy = gt_norm.reshape(4, 2) * np.array([[w, h]], dtype=np.float32)
                out["iou"] = _polygon_iou(pred_xy, gt_xy)
                err_mean, err_max = _corner_errors_best_rotation(pred_xy, gt_xy)
                out["err_mean_px"] = err_mean
                out["err_max_px"] = err_max

                # Potential label order fix suggestion
                rot, rot_err = _best_gt_rotation_to_match_pred(gt_xy, pred_xy)
                out["gt_rot_to_teacher"] = int(rot)
                out["_gt_rot_err_px"] = rot_err  # internal (not written to CSV unless needed)

    except Exception:
        return out

    return out


def _read_split_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text().splitlines():
        name = raw.strip()
        if not name:
            continue
        # Normalize possible prefixes
        if name.startswith("images/"):
            name = name[len("images/") :]
        elif name.startswith("images-negative/"):
            name = name[len("images-negative/") :]
        lines.append(name)
    return lines


def _write_split(path: Path, lines: Iterable[str]) -> None:
    path.write_text("".join(f"{l}\n" for l in lines))


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_args():
    p = argparse.ArgumentParser(
        description="Clean doc-scanner-dataset-rev-new using a teacher detector as a validator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new",
        help="Dataset root (contains images/, labels/, train.txt, val.txt, images-negative/, labels-negative/).",
    )
    p.add_argument(
        "--teacher_repo",
        type=str,
        default="vendor/teacher-detector",
        help="Path to the teacher detector repo (for document_detector.py import).",
    )
    p.add_argument(
        "--backbone",
        type=str,
        default="vendor/teacher-detector/models/document_detector_backbone.tflite",
        help="Teacher backbone .tflite",
    )
    p.add_argument(
        "--head",
        type=str,
        default="vendor/teacher-detector/models/document_detector_head.tflite",
        help="Teacher head .tflite",
    )
    p.add_argument("--splits", type=str, default="train,val", help="Comma-separated split basenames (train,val)")
    p.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 4)), help="Process workers")
    p.add_argument(
        "--csv_out",
        type=str,
        default="teacher_vs_gt_full.csv",
        help="Output CSV name (written under data_root).",
    )
    p.add_argument(
        "--update_splits",
        action="store_true",
        help="Backup + replace train.txt/val.txt with cleaned versions.",
    )
    p.add_argument(
        "--fix_label_order",
        action="store_true",
        help="Rewrite label files to match teacher corner order when safe (backup originals).",
    )
    p.add_argument(
        "--remove_conf_high",
        type=float,
        default=0.95,
        help="Confidence threshold for considering teacher reliable for removals.",
    )
    p.add_argument(
        "--remove_iou_low",
        type=float,
        default=0.30,
        help="If (conf>=remove_conf_high AND iou<=remove_iou_low) -> remove (positives).",
    )
    p.add_argument(
        "--remove_conf_very_high",
        type=float,
        default=0.99,
        help="Very high confidence threshold (stricter).",
    )
    p.add_argument(
        "--remove_iou_med",
        type=float,
        default=0.50,
        help="If (conf>=remove_conf_very_high AND iou<=remove_iou_med) -> remove (positives).",
    )
    p.add_argument(
        "--remove_neg_conf",
        type=float,
        default=0.95,
        help="If negative sample and teacher detects with conf>=this -> remove from negatives.",
    )
    p.add_argument(
        "--label_fix_min_iou",
        type=float,
        default=0.98,
        help="Only suggest/apply label order fix when IoU >= this.",
    )
    p.add_argument(
        "--label_fix_max_err_px",
        type=float,
        default=10.0,
        help="Only suggest/apply label order fix when mean GT-vs-teacher corner err <= this.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]

    csv_path = data_root / str(args.csv_out)
    removed_pos_path = data_root / f"removed_positives_teacher_{_timestamp()}.txt"
    removed_neg_path = data_root / f"removed_negatives_teacher_{_timestamp()}.txt"
    label_fix_path = data_root / f"label_order_fixes_teacher_{_timestamp()}.csv"

    # Build tasks
    tasks: list[Task] = []
    split_lines: dict[str, list[str]] = {}
    for split in splits:
        split_file = data_root / f"{split}.txt"
        if not split_file.exists():
            raise SystemExit(f"Missing split file: {split_file}")
        lines = _read_split_lines(split_file)
        split_lines[split] = lines
        for name in lines:
            is_neg = name.startswith("negative_")
            tasks.append(Task(split=split, filename=name, is_negative=is_neg))

    print("\n" + "=" * 80)
    print("Teacher vs GT audit (full dataset)")
    print("=" * 80)
    print(f"Dataset: {data_root}")
    print(f"Splits: {splits}")
    print(f"Total samples (incl negatives): {len(tasks)}")

    # Run inference+comparison
    fieldnames = [
        "split",
        "filename",
        "is_negative",
        "image_exists",
        "label_exists",
        "teacher_status",
        "teacher_conf",
        "teacher_detected",
        "gt_valid",
        "gt_oob",
        "gt_area_ratio",
        "iou",
        "err_mean_px",
        "err_max_px",
        "gt_rot_to_teacher",
        "w",
        "h",
    ]

    removed_pos: set[str] = set()
    removed_neg: set[str] = set()
    label_fixes: dict[str, int] = {}

    # Ensure we don't accidentally mix old results
    if csv_path.exists():
        backup = csv_path.with_suffix(f".bak_{_timestamp()}.csv")
        csv_path.rename(backup)

    with csv_path.open("w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(
            max_workers=int(args.workers),
            initializer=_init_worker,
            initargs=(args.teacher_repo, args.backbone, args.head, str(data_root)),
        ) as ex:
            processed = 0
            for row in tqdm(
                ex.map(_process_one, tasks, chunksize=32),
                total=len(tasks),
                desc="Teacher audit",
            ):
                processed += 1

                # Removal logic
                is_negative = bool(int(row.get("is_negative", 0)))
                image_exists = bool(int(row.get("image_exists", 0)))
                if image_exists:
                    teacher_detected = bool(int(row.get("teacher_detected", 0)))
                    conf = float(row.get("teacher_conf", 0.0) or 0.0)

                    if is_negative:
                        if teacher_detected and conf >= float(args.remove_neg_conf):
                            removed_neg.add(str(row["filename"]))
                    else:
                        gt_valid = bool(int(row.get("gt_valid", 0)))
                        gt_oob = bool(int(row.get("gt_oob", 0)))
                        iou = row.get("iou", "")
                        iou_val = float(iou) if iou != "" else None

                        # If GT is missing/invalid, remove.
                        if not gt_valid:
                            removed_pos.add(str(row["filename"]))
                        # Clear out-of-bounds is very likely wrong.
                        elif gt_oob:
                            removed_pos.add(str(row["filename"]))
                        # Strong mismatch with high teacher confidence.
                        elif teacher_detected and iou_val is not None:
                            if (conf >= float(args.remove_conf_very_high) and iou_val <= float(args.remove_iou_med)) or (
                                conf >= float(args.remove_conf_high) and iou_val <= float(args.remove_iou_low)
                            ):
                                removed_pos.add(str(row["filename"]))

                        # Label order fix suggestion
                        if (
                            not is_negative
                            and not gt_oob
                            and gt_valid
                            and teacher_detected
                            and iou_val is not None
                            and iou_val >= float(args.label_fix_min_iou)
                        ):
                            rot = row.get("gt_rot_to_teacher", "")
                            err = row.get("_gt_rot_err_px", None)
                            if rot != "" and err is not None:
                                rot_i = int(rot)
                                if rot_i != 0 and float(err) <= float(args.label_fix_max_err_px):
                                    label_fixes[str(row["filename"])] = rot_i

                # Write row (omit internal keys)
                for k in list(row.keys()):
                    if k.startswith("_"):
                        row.pop(k, None)
                writer.writerow({k: row.get(k, "") for k in fieldnames})
                if processed % 2000 == 0:
                    fcsv.flush()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"CSV: {csv_path}")
    print(f"Removed positives: {len(removed_pos)}")
    print(f"Removed negatives: {len(removed_neg)}")
    print(f"Label order fixes suggested: {len(label_fixes)}")

    removed_pos_path.write_text("".join(f"{n}\n" for n in sorted(removed_pos)))
    removed_neg_path.write_text("".join(f"{n}\n" for n in sorted(removed_neg)))
    print(f"Removed positives list: {removed_pos_path}")
    print(f"Removed negatives list: {removed_neg_path}")

    if args.fix_label_order and label_fixes:
        # Backup only the files we actually modify.
        backup_dir = data_root / f"labels_backup_teacher_orderfix_{_timestamp()}"
        (backup_dir / "labels").mkdir(parents=True, exist_ok=True)
        fixes = []

        for filename, rot in tqdm(label_fixes.items(), desc="Fixing label order"):
            label_path = data_root / "labels" / (Path(filename).stem + ".txt")
            if not label_path.exists():
                continue
            orig = label_path.read_text()
            (backup_dir / "labels" / label_path.name).write_text(orig)
            coords = _load_yolo_polygon(label_path)
            if coords is None:
                continue
            pts = coords.reshape(4, 2)
            pts_fixed = np.roll(pts, -int(rot), axis=0).reshape(-1)
            fixed_line = "0 " + " ".join(f"{float(x):.6f}" for x in pts_fixed.tolist()) + "\n"
            label_path.write_text(fixed_line)
            fixes.append({"filename": filename, "rot": int(rot)})

        with label_fix_path.open("w", newline="") as ffix:
            w = csv.DictWriter(ffix, fieldnames=["filename", "rot"])
            w.writeheader()
            w.writerows(fixes)
        print(f"Label order fixes applied: {len(fixes)}")
        print(f"Label backup dir: {backup_dir}")
        print(f"Fix log: {label_fix_path}")

    # Write cleaned splits
    cleaned_splits = {}
    for split in splits:
        lines = split_lines[split]
        cleaned = []
        for name in lines:
            is_neg = name.startswith("negative_")
            if is_neg:
                if name in removed_neg:
                    continue
                cleaned.append(name)
            else:
                if name in removed_pos:
                    continue
                cleaned.append(name)
        cleaned_splits[split] = cleaned

    # Save cleaned variants (always)
    for split in splits:
        out_split = data_root / f"{split}_teacher_cleaned.txt"
        _write_split(out_split, cleaned_splits[split])
        print(f"Cleaned split written: {out_split} ({len(cleaned_splits[split])} lines)")

    # Optionally update train.txt/val.txt in-place (with backups)
    if args.update_splits:
        stamp = _timestamp()
        for split in splits:
            orig_path = data_root / f"{split}.txt"
            backup_path = data_root / f"{split}.orig_before_teacher_clean_{stamp}.txt"
            backup_path.write_text(orig_path.read_text())
            orig_path.write_text((data_root / f"{split}_teacher_cleaned.txt").read_text())
            print(f"Updated {orig_path} (backup: {backup_path})")

    print("\nDone.")


if __name__ == "__main__":
    main()
