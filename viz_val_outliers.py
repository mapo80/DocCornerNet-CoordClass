"""
Visualize hard-tail samples by rendering GT vs model predictions.

This script focuses on a split file (default: val_outliers) and produces:
- One composite image with GT + both model predictions
- Per-model images (GT + one prediction)
- A JSON report with per-image metrics (IoU, corner error, score)
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

from dataset import IMAGENET_MEAN, IMAGENET_STD, load_split_file, load_yolo_label
from metrics import compute_polygon_iou, compute_corner_error
from model import create_model


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    img_size: int
    input_norm: str
    backbone_include_preprocessing: bool


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render GT vs predictions for val_outliers (or any custom split).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Dataset root containing images/, labels/, and split files.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="val_outliers",
        help="Split name (uses <split>_with_negative_v2.txt, then fallbacks).",
    )
    p.add_argument(
        "--model_with_outliers",
        type=str,
        required=True,
        help="Checkpoint dir / .keras / weights for the model trained with outliers.",
    )
    p.add_argument(
        "--model_without_outliers",
        type=str,
        required=True,
        help="Checkpoint dir / .keras / weights for the model trained without outliers.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="./outliers_tail/val_outliers",
        help="Output directory where images and report.json are saved.",
    )
    p.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="If >0, process only the first N images (debug).",
    )
    return p.parse_args()


def _find_config_path(model_path: Path) -> Optional[Path]:
    candidates: List[Path] = []
    if model_path.is_dir():
        candidates.extend([model_path / "config.json", model_path.parent / "config.json"])
    else:
        candidates.extend([model_path.parent / "config.json"])
    for c in candidates:
        if c.exists():
            return c
    return None


def _normalize_backbone_weights(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None
    return value


def _load_model_and_spec(model_path_str: str, name: str) -> Tuple[tf.keras.Model, ModelSpec]:
    model_path = Path(model_path_str)
    config_path = _find_config_path(model_path)
    if not config_path:
        raise FileNotFoundError(f"Missing config.json for {model_path} (needed for visualization).")
    cfg = json.loads(config_path.read_text())

    backbone = cfg["backbone"]
    alpha = float(cfg["alpha"])
    fpn_ch = int(cfg["fpn_ch"])
    simcc_ch = int(cfg["simcc_ch"])
    img_size = int(cfg["img_size"])
    num_bins = int(cfg.get("num_bins", img_size))
    tau = float(cfg.get("tau", 1.0))
    input_norm = str(cfg.get("input_norm", "imagenet"))
    backbone_include_preprocessing = bool(cfg.get("backbone_include_preprocessing", False))
    backbone_minimalistic = bool(cfg.get("backbone_minimalistic", False))
    backbone_weights = _normalize_backbone_weights(cfg.get("backbone_weights", None))

    model = create_model(
        backbone=backbone,
        alpha=alpha,
        backbone_minimalistic=backbone_minimalistic,
        backbone_include_preprocessing=backbone_include_preprocessing,
        backbone_weights=backbone_weights,
        fpn_ch=fpn_ch,
        simcc_ch=simcc_ch,
        img_size=img_size,
        num_bins=num_bins,
        tau=tau,
    )

    # Prefer weights.h5 inside directory, otherwise allow direct weights path.
    if model_path.is_dir():
        weights_path = model_path / "best_model.weights.h5"
        if not weights_path.exists():
            weights_path = model_path / "final_model.weights.h5"
        if not weights_path.exists():
            raise FileNotFoundError(f"No weights found in {model_path} (expected best_model.weights.h5).")
    else:
        weights_path = model_path

    model.load_weights(str(weights_path))

    spec = ModelSpec(
        name=name,
        path=model_path,
        img_size=img_size,
        input_norm=input_norm,
        backbone_include_preprocessing=backbone_include_preprocessing,
    )
    return model, spec


def _find_split_file(data_root: Path, split: str) -> Path:
    for candidate in [
        data_root / f"{split}_with_negative_v2.txt",
        data_root / f"{split}_with_negative.txt",
        data_root / f"{split}.txt",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No split file found for '{split}' in {data_root}")


def _find_image_path(data_root: Path, image_name: str, negative_dir: str = "images-negative") -> Path:
    if image_name.startswith("negative_"):
        base = data_root / negative_dir / image_name
    else:
        base = data_root / "images" / image_name
    if base.exists():
        return base

    stem = Path(image_name).stem
    parent = base.parent
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        candidate = parent / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return base


def _normalize_image(np_img_raw255: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower().strip()
    x = np_img_raw255.astype(np.float32)
    if mode == "imagenet":
        x = x / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        x = (x - mean) / std
        return x
    if mode in {"zero_one", "0_1", "01"}:
        return x / 255.0
    if mode in {"raw255", "0_255", "0255"}:
        return x
    raise ValueError(f"Unsupported input_norm='{mode}'.")


def _coords_to_bbox_px(coords_norm: np.ndarray, img_size: int) -> Tuple[int, int, int, int]:
    coords_px = coords_norm.reshape(4, 2) * float(img_size)
    xs = coords_px[:, 0]
    ys = coords_px[:, 1]
    x1 = int(np.clip(xs.min(), 0, img_size - 1))
    y1 = int(np.clip(ys.min(), 0, img_size - 1))
    x2 = int(np.clip(xs.max(), 0, img_size - 1))
    y2 = int(np.clip(ys.max(), 0, img_size - 1))
    return x1, y1, x2, y2


def _draw_overlay(
    img_rgb: Image.Image,
    img_size: int,
    gt: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    meta_text: str,
) -> Image.Image:
    img = img_rgb.copy()
    draw = ImageDraw.Draw(img)
    w = max(2, img_size // 128)

    def draw_quad(coords_norm: np.ndarray, color: Tuple[int, int, int]):
        pts = (coords_norm.reshape(4, 2) * float(img_size)).tolist()
        pts = [(float(x), float(y)) for x, y in pts]
        draw.line(pts + [pts[0]], fill=color, width=w)

    def draw_corners(coords_norm: np.ndarray, color: Tuple[int, int, int]):
        pts = (coords_norm.reshape(4, 2) * float(img_size)).tolist()
        r = max(2, img_size // 96)
        for i, (x, y) in enumerate(pts):
            x0, y0 = int(x) - r, int(y) - r
            x1, y1 = int(x) + r, int(y) + r
            draw.ellipse([x0, y0, x1, y1], outline=color, width=max(1, w))
            draw.text((int(x) + r + 1, int(y) - r), str(i), fill=color)

    # GT in green, with-outliers in red, without-outliers in blue.
    draw_quad(gt, (0, 255, 0))
    draw_corners(gt, (0, 255, 0))

    draw_quad(pred_a, (255, 0, 0))
    draw_corners(pred_a, (255, 0, 0))

    draw_quad(pred_b, (0, 128, 255))
    draw_corners(pred_b, (0, 128, 255))

    draw.rectangle([0, 0, img_size - 1, 22], fill=(0, 0, 0))
    draw.text((4, 4), meta_text, fill=(255, 255, 255))
    return img


def _draw_single(
    img_rgb: Image.Image,
    img_size: int,
    gt: np.ndarray,
    pred: np.ndarray,
    pred_label: str,
    meta_text: str,
) -> Image.Image:
    img = img_rgb.copy()
    draw = ImageDraw.Draw(img)
    w = max(2, img_size // 128)

    # GT
    gt_pts = (gt.reshape(4, 2) * float(img_size)).tolist()
    gt_pts = [(float(x), float(y)) for x, y in gt_pts]
    draw.line(gt_pts + [gt_pts[0]], fill=(0, 255, 0), width=w)
    r = max(2, img_size // 96)
    for i, (x, y) in enumerate(gt_pts):
        x0, y0 = int(x) - r, int(y) - r
        x1, y1 = int(x) + r, int(y) + r
        draw.ellipse([x0, y0, x1, y1], outline=(0, 255, 0), width=max(1, w))
        draw.text((int(x) + r + 1, int(y) - r), str(i), fill=(0, 255, 0))

    # Pred
    pr_pts = (pred.reshape(4, 2) * float(img_size)).tolist()
    pr_pts = [(float(x), float(y)) for x, y in pr_pts]
    draw.line(pr_pts + [pr_pts[0]], fill=(255, 0, 0), width=w)
    for i, (x, y) in enumerate(pr_pts):
        x0, y0 = int(x) - r, int(y) - r
        x1, y1 = int(x) + r, int(y) + r
        draw.ellipse([x0, y0, x1, y1], outline=(255, 0, 0), width=max(1, w))
        draw.text((int(x) + r + 1, int(y) - r), str(i), fill=(255, 0, 0))
    draw.text((4, 24), pred_label, fill=(255, 0, 0))

    draw.rectangle([0, 0, img_size - 1, 22], fill=(0, 0, 0))
    draw.text((4, 4), meta_text, fill=(255, 255, 255))
    return img


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root)

    model_with, spec_with = _load_model_and_spec(args.model_with_outliers, "with_outliers")
    model_without, spec_without = _load_model_and_spec(args.model_without_outliers, "no_outliers")

    if spec_with.img_size != spec_without.img_size:
        raise ValueError(
            f"Different img_size: with_outliers={spec_with.img_size}, no_outliers={spec_without.img_size}. "
            "This script currently expects identical sizes."
        )
    if spec_with.input_norm != spec_without.input_norm:
        raise ValueError(
            f"Different input_norm: with_outliers={spec_with.input_norm}, no_outliers={spec_without.input_norm}. "
            "This script expects identical normalization."
        )

    img_size = spec_with.img_size
    split_file = _find_split_file(data_root, args.split)
    names = load_split_file(str(split_file))
    if args.max_images > 0:
        names = names[: args.max_images]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare").mkdir(parents=True, exist_ok=True)
    (out_dir / "with_outliers").mkdir(parents=True, exist_ok=True)
    (out_dir / "no_outliers").mkdir(parents=True, exist_ok=True)

    report: Dict[str, dict] = {
        "data_root": str(data_root),
        "split": args.split,
        "split_file": str(split_file),
        "count": len(names),
        "models": {
            "with_outliers": {"path": str(spec_with.path), "img_size": img_size, "input_norm": spec_with.input_norm},
            "no_outliers": {"path": str(spec_without.path), "img_size": img_size, "input_norm": spec_without.input_norm},
        },
        "items": [],
    }

    for idx, image_name in enumerate(names):
        img_path = _find_image_path(data_root, image_name)
        label_path = data_root / "labels" / f"{Path(image_name).stem}.txt"
        gt_coords, has_doc = load_yolo_label(str(label_path))
        if not has_doc:
            continue

        pil = Image.open(img_path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        np_raw = np.array(pil, dtype=np.float32)  # 0..255

        x = _normalize_image(np_raw, spec_with.input_norm)
        x = np.expand_dims(x, axis=0).astype(np.float32)

        out_with = model_with(x, training=False)
        out_without = model_without(x, training=False)

        def unpack(outputs):
            if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
                coords_pred = outputs[0].numpy()[0]
                score_logit = float(outputs[1].numpy().reshape(-1)[0])
            else:
                coords_pred = outputs["coords"].numpy()[0]
                score_logit = float(outputs["score_logit"].numpy().reshape(-1)[0])
            score = float(1.0 / (1.0 + np.exp(-np.clip(score_logit, -60.0, 60.0))))
            return np.clip(coords_pred.astype(np.float32), 0.0, 1.0), score

        pred_with, score_with = unpack(out_with)
        pred_without, score_without = unpack(out_without)

        iou_with = float(compute_polygon_iou(pred_with, gt_coords))
        iou_without = float(compute_polygon_iou(pred_without, gt_coords))
        ce_with, _ = compute_corner_error(pred_with, gt_coords, img_size=img_size)
        ce_without, _ = compute_corner_error(pred_without, gt_coords, img_size=img_size)

        meta = (
            f"{idx+1}/{len(names)} {Path(image_name).name} | "
            f"IoU w={iou_with:.3f} n={iou_without:.3f} | "
            f"err w={ce_with:.2f}px n={ce_without:.2f}px"
        )

        img_compare = _draw_overlay(
            pil,
            img_size=img_size,
            gt=gt_coords,
            pred_a=pred_with,
            pred_b=pred_without,
            meta_text=meta,
        )
        img_with = _draw_single(
            pil,
            img_size=img_size,
            gt=gt_coords,
            pred=pred_with,
            pred_label="with_outliers",
            meta_text=f"{Path(image_name).name} | IoU {iou_with:.3f} | err {ce_with:.2f}px | score {score_with:.3f}",
        )
        img_without = _draw_single(
            pil,
            img_size=img_size,
            gt=gt_coords,
            pred=pred_without,
            pred_label="no_outliers",
            meta_text=f"{Path(image_name).name} | IoU {iou_without:.3f} | err {ce_without:.2f}px | score {score_without:.3f}",
        )

        stem = Path(image_name).stem
        img_compare.save(out_dir / "compare" / f"{idx:02d}_{stem}.jpg", quality=95)
        img_with.save(out_dir / "with_outliers" / f"{idx:02d}_{stem}.jpg", quality=95)
        img_without.save(out_dir / "no_outliers" / f"{idx:02d}_{stem}.jpg", quality=95)

        report["items"].append(
            {
                "image_name": image_name,
                "image_path": str(img_path),
                "label_path": str(label_path),
                "with_outliers": {"iou": iou_with, "corner_err_px": float(ce_with), "score": score_with},
                "no_outliers": {"iou": iou_without, "corner_err_px": float(ce_without), "score": score_without},
            }
        )

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Saved {len(report['items'])} visualizations to: {out_dir}")
    print(f"Report: {out_dir/'report.json'}")


if __name__ == "__main__":
    main()
