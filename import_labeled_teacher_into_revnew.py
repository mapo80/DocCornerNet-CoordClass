#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any


def _read_lines(path: Path) -> list[str]:
    items: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            items.append(s)
    return items


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=str(path.parent), prefix=path.name, suffix=".tmp"
    ) as f:
        f.write(text)
        tmp = Path(f.name)
    tmp.replace(path)


def _write_yolo_poly_label(label_path: Path, coords_norm8: list[float]) -> None:
    if len(coords_norm8) != 8:
        raise ValueError(f"Expected 8 coords, got {len(coords_norm8)}")
    parts = ["0"] + [f"{float(v):.6f}" for v in coords_norm8]
    _atomic_write_text(label_path, " ".join(parts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates_txt",
        type=Path,
        default=Path(
            "/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-labeled/new_not_in_revnew_iou_ge097_no_midv_smartdoc.txt"
        ),
    )
    parser.add_argument(
        "--teacher_json",
        type=Path,
        default=Path(
            "/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-labeled/teacher_vs_gt_labeled.json"
        ),
    )
    parser.add_argument(
        "--labeled_images",
        type=Path,
        default=Path(
            "/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-labeled/images"
        ),
    )
    parser.add_argument(
        "--revnew_images",
        type=Path,
        default=Path(
            "/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new/images"
        ),
    )
    parser.add_argument(
        "--revnew_labels",
        type=Path,
        default=Path(
            "/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new/labels"
        ),
    )
    parser.add_argument("--overwrite_images", action="store_true")
    parser.add_argument("--overwrite_labels", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--report_json",
        type=Path,
        default=Path(
            "/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new/import_labeled_teacher_report.json"
        ),
    )
    args = parser.parse_args()

    for p in [args.candidates_txt, args.teacher_json]:
        if not p.exists():
            raise SystemExit(f"Missing file: {p}")
    for p in [args.labeled_images, args.revnew_images, args.revnew_labels]:
        if not p.exists():
            raise SystemExit(f"Missing dir: {p}")

    candidates = _read_lines(args.candidates_txt)
    with args.teacher_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    samples: list[dict[str, Any]] = data.get("samples", [])
    by_name: dict[str, dict[str, Any]] = {s.get("filename"): s for s in samples if s.get("filename")}

    report: dict[str, Any] = {
        "inputs": {
            "candidates_txt": str(args.candidates_txt),
            "teacher_json": str(args.teacher_json),
            "labeled_images": str(args.labeled_images),
            "revnew_images": str(args.revnew_images),
            "revnew_labels": str(args.revnew_labels),
        },
        "flags": {
            "overwrite_images": bool(args.overwrite_images),
            "overwrite_labels": bool(args.overwrite_labels),
            "dry_run": bool(args.dry_run),
        },
        "counts": {
            "candidates": len(candidates),
            "copied_images": 0,
            "skipped_images_exists": 0,
            "written_labels": 0,
            "skipped_labels_exists": 0,
            "missing_source_image": 0,
            "missing_teacher_pred": 0,
            "teacher_not_ok": 0,
        },
        "missing_source_image": [],
        "missing_teacher_pred": [],
        "teacher_not_ok": [],
    }

    for filename in candidates:
        src_img = args.labeled_images / filename
        dst_img = args.revnew_images / filename
        dst_lbl = args.revnew_labels / f"{Path(filename).stem}.txt"

        if not src_img.exists():
            report["counts"]["missing_source_image"] += 1
            report["missing_source_image"].append(filename)
            continue

        sample = by_name.get(filename)
        if sample is None:
            report["counts"]["missing_teacher_pred"] += 1
            report["missing_teacher_pred"].append(filename)
            continue
        if sample.get("detector_status") != "OK":
            report["counts"]["teacher_not_ok"] += 1
            report["teacher_not_ok"].append(filename)
            continue
        pred = sample.get("pred_norm8")
        if not isinstance(pred, list) or len(pred) != 8:
            report["counts"]["missing_teacher_pred"] += 1
            report["missing_teacher_pred"].append(filename)
            continue

        # Copy image
        if dst_img.exists() and not args.overwrite_images:
            report["counts"]["skipped_images_exists"] += 1
        else:
            if not args.dry_run:
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_img, dst_img)
            report["counts"]["copied_images"] += 1

        # Write label
        if dst_lbl.exists() and not args.overwrite_labels:
            report["counts"]["skipped_labels_exists"] += 1
        else:
            if not args.dry_run:
                _write_yolo_poly_label(dst_lbl, [float(x) for x in pred])
            report["counts"]["written_labels"] += 1

    if not args.dry_run:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Done.")
    print("Candidates:", report["counts"]["candidates"])
    print("Copied images:", report["counts"]["copied_images"])
    print("Written labels:", report["counts"]["written_labels"])
    print("Report:", args.report_json)


if __name__ == "__main__":
    main()
