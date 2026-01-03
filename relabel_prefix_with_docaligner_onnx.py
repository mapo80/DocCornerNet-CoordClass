#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _read_split(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _order_points_tl_tr_br_bl(coords8: np.ndarray) -> np.ndarray:
    pts = coords8.reshape(4, 2).astype(np.float32)
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts_ccw = pts[np.argsort(angles)]
    start = int(np.argmin(pts_ccw[:, 0] + pts_ccw[:, 1]))
    pts_ccw = np.roll(pts_ccw, -start, axis=0)
    if pts_ccw[1, 1] > pts_ccw[-1, 1]:
        pts_ccw = np.vstack([pts_ccw[0:1], pts_ccw[:0:-1]])
    return pts_ccw.reshape(-1)


def _preprocess_to_chw_256(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((256, 256), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32)  # HWC
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = x / 255.0
    return x


def _coords_from_heatmap(
    hm: np.ndarray,
    heatmap_threshold: float,
    w: int,
    h: int,
) -> tuple[bool, float, float, float]:
    if hm.ndim != 2:
        return False, 0.0, -1.0, -1.0
    maxv = float(np.max(hm))
    if not np.isfinite(maxv) or maxv < float(heatmap_threshold):
        return False, maxv, -1.0, -1.0

    mask = hm >= float(heatmap_threshold)
    if not np.any(mask):
        return False, maxv, -1.0, -1.0

    ys, xs = np.nonzero(mask)
    weights = hm[mask].astype(np.float64)
    wsum = float(weights.sum())
    if wsum <= 0.0 or not np.isfinite(wsum):
        return False, maxv, -1.0, -1.0

    x_hm = float((xs.astype(np.float64) * weights).sum() / wsum)
    y_hm = float((ys.astype(np.float64) * weights).sum() / wsum)

    hm_h, hm_w = int(hm.shape[0]), int(hm.shape[1])
    x_px = (x_hm / max(1.0, float(hm_w - 1))) * max(1.0, float(w - 1))
    y_px = (y_hm / max(1.0, float(hm_h - 1))) * max(1.0, float(h - 1))
    x_norm = float(np.clip(x_px / float(w), 0.0, 1.0))
    y_norm = float(np.clip(y_px / float(h), 0.0, 1.0))
    return True, maxv, x_norm, y_norm


@dataclass(frozen=True)
class Item:
    split: str
    name: str


def _write_yolo_poly(label_path: Path, coords8: np.ndarray | None) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    if coords8 is None:
        label_path.write_text("", encoding="utf-8")
        return
    parts = ["0"] + [f"{float(v):.6f}" for v in coords8.tolist()]
    label_path.write_text(" ".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Relabel a subset of dataset images with DocAligner heatmap ONNX (writes YOLO poly labels).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_root",
        type=Path,
        default=Path(
            "/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-last"
        ),
    )
    p.add_argument(
        "--onnx_path",
        type=Path,
        default=Path(
            "/Volumes/ZX20/ML-Models/DocScannerDetection/third-party/DocAligner/docaligner/heatmap_reg/ckpt/fastvit_sa24_h_e_bifpn_256_fp32.onnx"
        ),
    )
    p.add_argument("--splits", type=str, default="train,val", help="Comma-separated split basenames")
    p.add_argument("--prefixes", type=str, default="smartdoc", help="Comma-separated filename prefixes to relabel")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="onnxruntime provider preference")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--heatmap_threshold", type=float, default=0.3)
    p.add_argument(
        "--backup",
        action="store_true",
        help="Backup existing labels to labels_backup_<timestamp>/ before overwriting",
    )
    p.add_argument(
        "--only_if_empty",
        action="store_true",
        help="Only relabel if current label file is empty/missing (safer).",
    )
    p.add_argument(
        "--report_json",
        type=Path,
        default=None,
        help="Defaults to <data_root>/docaligner_relabel_<timestamp>.json",
    )
    args = p.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    data_root: Path = args.data_root
    onnx_path: Path = args.onnx_path
    if not onnx_path.exists():
        raise SystemExit(f"Missing ONNX: {onnx_path}")

    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    prefixes = [x.strip().lower() for x in str(args.prefixes).split(",") if x.strip()]

    images_dir = data_root / "images"
    labels_dir = data_root / "labels"
    for d in [images_dir, labels_dir]:
        if not d.exists():
            raise SystemExit(f"Missing dir: {d}")

    try:
        import onnxruntime as ort
    except Exception as e:
        raise SystemExit("onnxruntime is required (pip install onnxruntime)") from e

    if str(args.device).lower() == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)
    input_name = sess.get_inputs()[0].name

    items: list[Item] = []
    for split in splits:
        split_path = data_root / f"{split}.txt"
        if not split_path.exists():
            raise SystemExit(f"Missing split file: {split_path}")
        for name in _read_split(split_path):
            low = name.lower()
            if any(low.startswith(pref) for pref in prefixes):
                items.append(Item(split=split, name=name))

    if not items:
        raise SystemExit(f"No items matched prefixes={prefixes} in splits={splits}")

    ts = _timestamp()
    if args.report_json is None:
        report_path = data_root / f"docaligner_relabel_{ts}.json"
    else:
        report_path = args.report_json

    backup_dir: Path | None = None
    if args.backup:
        backup_dir = data_root / f"labels_backup_{ts}_before_docaligner"
        backup_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "data_root": str(data_root),
        "onnx_path": str(onnx_path),
        "providers": sess.get_providers(),
        "params": {
            "splits": splits,
            "prefixes": prefixes,
            "batch_size": int(args.batch_size),
            "heatmap_threshold": float(args.heatmap_threshold),
            "only_if_empty": bool(args.only_if_empty),
            "backup": bool(args.backup),
        },
        "counts": {
            "matched": len(items),
            "processed": 0,
            "relabelled": 0,
            "skipped_nonempty": 0,
            "detected": 0,
            "not_detected": 0,
            "errors": 0,
        },
        "not_detected": [],
        "errors": [],
        "confidence": {"mean": 0.0, "p50": 0.0, "p95": 0.0},
    }

    confs: list[float] = []

    batch_x: list[np.ndarray] = []
    batch_meta: list[tuple[Item, int, int, Path]] = []

    def flush() -> None:
        if not batch_meta:
            return
        x = np.stack(batch_x, axis=0).astype(np.float32, copy=False)  # [B,3,256,256]
        heat = sess.run(None, {input_name: x})[0]  # [B,4,128,128]
        heat = np.asarray(heat, dtype=np.float32)
        for i, (it, w, h, label_path) in enumerate(batch_meta):
            pts: list[float] = []
            cmax: list[float] = []
            ok_all = True
            for ch in range(4):
                ok, maxv, x_n, y_n = _coords_from_heatmap(
                    heat[i, ch], heatmap_threshold=float(args.heatmap_threshold), w=int(w), h=int(h)
                )
                if not ok:
                    ok_all = False
                    break
                cmax.append(float(maxv))
                pts.extend([float(x_n), float(y_n)])

            report["counts"]["processed"] += 1

            if not ok_all:
                report["counts"]["not_detected"] += 1
                report["not_detected"].append(it.name)
                # Keep label as-is if only_if_empty==False? We still "processed".
                continue

            coords = _order_points_tl_tr_br_bl(np.array(pts, dtype=np.float32))
            coords = np.clip(coords, 0.0, 1.0)
            conf = float(np.mean(cmax)) if cmax else 0.0
            confs.append(conf)

            _write_yolo_poly(label_path, coords)
            report["counts"]["relabelled"] += 1
            report["counts"]["detected"] += 1

        batch_x.clear()
        batch_meta.clear()

    iterator = tqdm(items, desc="DocAligner relabel") if tqdm else items
    for it in iterator:
        img_path = images_dir / it.name
        label_path = labels_dir / f"{Path(it.name).stem}.txt"

        try:
            if args.only_if_empty:
                if label_path.exists() and label_path.read_text(encoding="utf-8").strip():
                    report["counts"]["skipped_nonempty"] += 1
                    continue

            if backup_dir is not None:
                if label_path.exists():
                    (backup_dir / label_path.name).write_text(
                        label_path.read_text(encoding="utf-8"), encoding="utf-8"
                    )
                else:
                    (backup_dir / label_path.name).write_text("", encoding="utf-8")

            with Image.open(img_path) as img:
                img = img.convert("RGB")
                w, h = img.size
                x = _preprocess_to_chw_256(img)

            batch_x.append(x)
            batch_meta.append((it, int(w), int(h), label_path))

            if len(batch_meta) >= int(args.batch_size):
                flush()
        except Exception as e:
            report["counts"]["errors"] += 1
            report["errors"].append({"name": it.name, "error": str(e)})

    flush()

    # Mark undetected cases as such (processed already) and keep labels unchanged if they existed.
    report["counts"]["not_detected"] = len(report["not_detected"])
    report["counts"]["errors"] = len(report["errors"])

    if confs:
        arr = np.array(confs, dtype=np.float32)
        report["confidence"] = {
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Report:", report_path)
    print("Matched:", report["counts"]["matched"])
    print("Relabelled:", report["counts"]["relabelled"])
    print("Not detected:", report["counts"]["not_detected"])


if __name__ == "__main__":
    main()
