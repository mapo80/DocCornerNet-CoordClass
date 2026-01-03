#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from metrics import compute_polygon_iou

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: Pillow (pip install pillow)") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


@dataclass(frozen=True)
class Datasets:
    labeled_root: Path
    labeled_images: Path
    labeled_labels: Path
    revnew_root: Path
    revnew_train: Path
    revnew_val: Path


def _iter_filenames_from_txt(path: Path) -> set[str]:
    items: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            items.add(name)
    return items


def _load_revnew_filenames(datasets: Datasets) -> set[str]:
    return _iter_filenames_from_txt(datasets.revnew_train) | _iter_filenames_from_txt(
        datasets.revnew_val
    )


def _load_detector(detector_root: Path, verbose: bool):
    detector_root = detector_root.resolve()
    sys.path.insert(0, str(detector_root))
    try:
        from document_detector import DocumentDetector  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import DocumentDetector from {detector_root}/document_detector.py"
        ) from exc
    return DocumentDetector(verbose=verbose)


def _parse_label_file(label_path: Path) -> np.ndarray:
    """
    Parse label format:
      class_id x0 y0 x1 y1 x2 y2 x3 y3
    Coordinates are normalized [0..1].
    """
    line = label_path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    parts = line.split()
    if len(parts) != 9:
        raise ValueError(f"Expected 9 tokens, got {len(parts)}: {label_path}")
    coords = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    return _order_tl_tr_br_bl(coords.reshape(4, 2)).reshape(-1)


def _order_tl_tr_br_bl(points: np.ndarray) -> np.ndarray:
    """
    Robust-ish quad ordering (TL, TR, BR, BL) based on sum/diff heuristics.
    Expects shape [4,2] in (x,y) with coordinates in the same space.
    """
    if points.shape != (4, 2):
        raise ValueError(f"Expected points shape (4,2), got {points.shape}")
    pts = points.astype(np.float32, copy=False)
    s = pts.sum(axis=1)
    diff = (pts[:, 1] - pts[:, 0]).reshape(-1)
    tl = pts[int(np.argmin(s))]
    br = pts[int(np.argmax(s))]
    tr = pts[int(np.argmin(diff))]
    bl = pts[int(np.argmax(diff))]
    return np.stack([tl, tr, br, bl], axis=0)


def _corners_xy_to_norm8(corners_xy: list[tuple[float, float]], w: int, h: int) -> np.ndarray:
    pts = np.array([[x / w, y / h] for x, y in corners_xy], dtype=np.float32)
    pts = np.clip(pts, 0.0, 1.0)
    pts = _order_tl_tr_br_bl(pts)
    return pts.reshape(-1)


def _progress(iterable, total: int, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labeled_root",
        type=Path,
        default=Path(
            "/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-labeled"
        ),
    )
    parser.add_argument(
        "--revnew_root",
        type=Path,
        default=Path(
            "/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new"
        ),
    )
    parser.add_argument(
        "--revnew_train",
        type=Path,
        default=Path(
            "/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new/train_final_pruned.txt"
        ),
    )
    parser.add_argument(
        "--revnew_val",
        type=Path,
        default=Path(
            "/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new/val_final_pruned.txt"
        ),
    )
    parser.add_argument(
        "--detector_root",
        type=Path,
        default=Path("vendor/teacher-detector"),
        help="Path containing document_detector.py (extracted TFLite detector).",
    )
    parser.add_argument("--iou_threshold", type=float, default=0.97)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path(
            "/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-labeled/teacher_vs_gt_labeled.json"
        ),
    )
    parser.add_argument(
        "--output_new_txt",
        type=Path,
        default=Path(
            "/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-labeled/new_not_in_revnew_iou_ge097.txt"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    datasets = Datasets(
        labeled_root=args.labeled_root,
        labeled_images=args.labeled_root / "images",
        labeled_labels=args.labeled_root / "labels",
        revnew_root=args.revnew_root,
        revnew_train=args.revnew_train,
        revnew_val=args.revnew_val,
    )

    for p in [datasets.labeled_images, datasets.labeled_labels]:
        if not p.exists():
            raise SystemExit(f"Missing path: {p}")
    for p in [datasets.revnew_train, datasets.revnew_val]:
        if not p.exists():
            raise SystemExit(f"Missing split file: {p}")

    revnew_filenames = _load_revnew_filenames(datasets)
    detector = _load_detector(args.detector_root, verbose=args.verbose)

    image_paths = sorted(datasets.labeled_images.glob("*.jpg"))
    total = len(image_paths)
    if total == 0:
        raise SystemExit(f"No images found in {datasets.labeled_images}")

    samples: list[dict[str, Any]] = []
    new_candidates: list[str] = []

    counts = {
        "total_images": total,
        "in_revnew": 0,
        "not_in_revnew": 0,
        "detected_ok": 0,
        "no_document_detected": 0,
        "iou_ge_threshold": 0,
        "new_not_in_revnew_iou_ge_threshold": 0,
    }

    for img_path in _progress(image_paths, total=total, desc="Annotating"):
        filename = img_path.name
        label_path = datasets.labeled_labels / f"{img_path.stem}.txt"

        row: dict[str, Any] = {
            "filename": filename,
            "image_path": str(img_path),
            "label_path": str(label_path),
        }

        in_revnew = filename in revnew_filenames
        row["in_revnew"] = in_revnew
        counts["in_revnew" if in_revnew else "not_in_revnew"] += 1

        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            row["w"] = int(w)
            row["h"] = int(h)
        except Exception as exc:
            row["error"] = f"image_load_failed: {exc}"
            samples.append(row)
            continue

        try:
            gt_norm8 = _parse_label_file(label_path)
            row["gt_norm8"] = [float(x) for x in gt_norm8.tolist()]
        except Exception as exc:
            row["error"] = f"label_parse_failed: {exc}"
            samples.append(row)
            continue

        try:
            det = detector.detect(img)
        except Exception as exc:
            row["error"] = f"detector_failed: {exc}"
            samples.append(row)
            continue

        status = det.get("status", "UNKNOWN")
        row["detector_status"] = status
        row["detector_confidence"] = float(det.get("confidence", 0.0) or 0.0)

        corners = det.get("corners")
        if status == "OK" and corners is not None:
            counts["detected_ok"] += 1
            row["detector_corners_xy"] = [(float(x), float(y)) for (x, y) in corners]
            pred_norm8 = _corners_xy_to_norm8(row["detector_corners_xy"], w=w, h=h)
            row["pred_norm8"] = [float(x) for x in pred_norm8.tolist()]

            iou = float(compute_polygon_iou(pred_norm8, gt_norm8))
            row["iou"] = iou

            if iou >= args.iou_threshold:
                counts["iou_ge_threshold"] += 1
                if not in_revnew:
                    counts["new_not_in_revnew_iou_ge_threshold"] += 1
                    new_candidates.append(filename)
        else:
            counts["no_document_detected"] += 1
            row["pred_norm8"] = None
            row["iou"] = None

        samples.append(row)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_new_txt.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "datasets": {
            "labeled_root": str(datasets.labeled_root),
            "revnew_train": str(datasets.revnew_train),
            "revnew_val": str(datasets.revnew_val),
        },
        "params": {
            "iou_threshold": float(args.iou_threshold),
            "detector_root": str(args.detector_root),
        },
        "summary": counts,
        "samples": samples,
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_new_txt.write_text("\n".join(new_candidates) + ("\n" if new_candidates else ""), encoding="utf-8")

    print("Saved:", args.output_json)
    print("Saved:", args.output_new_txt)
    print("Summary:", json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
