#!/usr/bin/env python3
"""
Create collage pages for samples filtered by GT-vs-teacher IoU range, overlaying:
  - GT polygon (green)
  - Teacher polygon (red, if detected)

Inputs:
  - <data_root>/{train,val}_gt_vs_teacher_iou.csv  (from compute_gt_vs_teacher_iou.py)
  - <data_root>/{train,val}_teacher.txt            (from generate_teacher_gt_splits.py)
  - <data_root>/labels/<stem>.txt                  (YOLO polygon GT)
  - <data_root>/images/<filename>                  (RGB image)

Outputs:
  - <out_dir>/candidates.csv                       (filtered rows)
  - <out_dir>/candidates_{train,val}.txt           (filenames)
  - <out_dir>/collages/*.jpg|png                   (pages)
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFile
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _read_yolo_polygon(label_path: Path) -> Optional[np.ndarray]:
    if not label_path.exists():
        return None
    try:
        line = label_path.read_text().strip().splitlines()[0].strip()
    except Exception:
        return None
    if not line:
        return None
    parts = line.split()
    if len(parts) < 9 or parts[0] != "0":
        return None
    try:
        coords = np.array([float(x) for x in parts[1:9]], dtype=np.float32)
    except Exception:
        return None
    if coords.shape != (8,) or not np.all(np.isfinite(coords)):
        return None
    return coords


def _load_teacher_split_file(path: Path) -> dict[str, np.ndarray]:
    """
    Load <split>_teacher.txt created by generate_teacher_gt_splits.py.
    Returns filename -> coords_norm[8] for detected==1.
    """
    out: dict[str, np.ndarray] = {}
    with path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or len(parts) != 11:
                continue
            name = parts[0]
            try:
                detected = int(parts[1])
            except Exception:
                continue
            if detected != 1:
                continue
            try:
                coords = np.array([float(x) for x in parts[3:11]], dtype=np.float32)
            except Exception:
                continue
            if coords.shape != (8,) or np.any(coords < 0.0) or not np.all(np.isfinite(coords)):
                continue
            out[name] = coords
    return out


def _draw_quad_on_tile(
    draw: ImageDraw.ImageDraw,
    coords: np.ndarray,
    w: int,
    h: int,
    scale: float,
    offx: int,
    offy: int,
    color: tuple[int, int, int],
    width: int,
) -> None:
    pts = coords.reshape(4, 2)
    xy = []
    for x, y in pts:
        px = float(x) * float(w) * scale + float(offx)
        py = float(y) * float(h) * scale + float(offy)
        xy.append((px, py))
    xy.append(xy[0])
    draw.line(xy, fill=color, width=width, joint="curve")


@dataclass(frozen=True)
class Row:
    split: str
    filename: str
    iou: float
    teacher_detected: int
    teacher_conf: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create collages for samples in an IoU range (GT vs teacher).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new",
        help="Dataset root.",
    )
    p.add_argument("--splits", type=str, default="train,val", help="Comma-separated split basenames")
    p.add_argument("--iou_min", type=float, default=0.0, help="Inclusive minimum IoU")
    p.add_argument("--iou_max", type=float, default=0.5, help="Exclusive maximum IoU")
    p.add_argument("--max_images", type=int, default=0, help="If >0, limit total candidates (debug)")

    p.add_argument("--tiles_per_page", type=int, default=1500, help="Number of thumbnails per collage page")
    p.add_argument("--cols", type=int, default=50, help="Columns per collage page")
    p.add_argument("--tile_size", type=int, default=128, help="Thumbnail tile size (square)")
    p.add_argument("--collage_format", type=str, default="jpg", choices=["jpg", "png"], help="Collage image format")
    p.add_argument("--jpeg_quality", type=int, default=85, help="JPEG quality when collage_format=jpg")

    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory. If empty, uses evaluation_results/gt_vs_teacher_iou_<range>_<timestamp>.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("evaluation_results") / (
            f"gt_vs_teacher_iou_{float(args.iou_min):.2f}_{float(args.iou_max):.2f}_{_timestamp()}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "collages").mkdir(parents=True, exist_ok=True)

    # Load teacher coords per split
    teacher_coords_by_split: dict[str, dict[str, np.ndarray]] = {}
    for split in splits:
        teacher_file = data_root / f"{split}_teacher.txt"
        if not teacher_file.exists():
            raise SystemExit(f"Missing {teacher_file}. Run generate_teacher_gt_splits.py first.")
        teacher_coords_by_split[split] = _load_teacher_split_file(teacher_file)

    # Load IoU CSVs and filter
    candidates: list[Row] = []
    for split in splits:
        p = data_root / f"{split}_gt_vs_teacher_iou.csv"
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run compute_gt_vs_teacher_iou.py first.")
        with p.open() as f:
            r = csv.DictReader(f)
            for row in r:
                fn = str(row.get("filename", "")).strip()
                if not fn:
                    continue
                try:
                    iou_s = row.get("iou", "")
                    if iou_s is None or str(iou_s).strip() == "":
                        continue
                    iou = float(iou_s)
                except Exception:
                    continue
                if not (float(args.iou_min) <= iou < float(args.iou_max)):
                    continue
                try:
                    teacher_det = int(float(row.get("teacher_detected", "0") or 0))
                except Exception:
                    teacher_det = 0
                try:
                    teacher_conf = float(row.get("teacher_conf", "0") or 0.0)
                except Exception:
                    teacher_conf = 0.0
                candidates.append(
                    Row(
                        split=split,
                        filename=fn,
                        iou=float(iou),
                        teacher_detected=teacher_det,
                        teacher_conf=teacher_conf,
                    )
                )

    # Deterministic sort: worst IoU first, then undetected, then name.
    candidates.sort(key=lambda x: (x.iou, 0 if x.teacher_detected == 0 else 1, x.filename))

    if args.max_images and args.max_images > 0:
        candidates = candidates[: int(args.max_images)]

    # Write candidate lists and a small CSV
    (out_dir / "candidates_train.txt").write_text(
        "".join(f"{r.filename}\n" for r in candidates if r.split == "train")
    )
    (out_dir / "candidates_val.txt").write_text("".join(f"{r.filename}\n" for r in candidates if r.split == "val"))
    fields = ["split", "filename", "iou", "teacher_detected", "teacher_conf"]
    with (out_dir / "candidates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in candidates:
            w.writerow({k: getattr(r, k) for k in fields})

    if not candidates:
        print(f"No candidates for IoU in [{args.iou_min},{args.iou_max}). Output: {out_dir}")
        return

    tiles_per_page = int(args.tiles_per_page)
    cols = int(args.cols)
    tile_size = int(args.tile_size)
    rows_per_page = int(math.ceil(float(tiles_per_page) / float(cols)))
    page_w = cols * tile_size
    page_h = rows_per_page * tile_size
    pages = int(math.ceil(len(candidates) / float(tiles_per_page)))

    print("\n" + "=" * 80)
    print("GT vs teacher IoU collage")
    print("=" * 80)
    print(f"Dataset: {data_root}")
    print(f"Splits: {splits}")
    print(f"IoU range: [{float(args.iou_min):.2f}, {float(args.iou_max):.2f})")
    print(f"Candidates: {len(candidates)}")
    print(f"Output: {out_dir}")

    for pidx in tqdm(range(pages), desc="Collage pages"):
        start = pidx * tiles_per_page
        end = min(len(candidates), start + tiles_per_page)
        page_items = candidates[start:end]

        collage = Image.new("RGB", (page_w, page_h), (18, 18, 18))
        for i, item in enumerate(page_items):
            col = i % cols
            row = i // cols
            ox = col * tile_size
            oy = row * tile_size

            tile = Image.new("RGB", (tile_size, tile_size), (18, 18, 18))
            draw = ImageDraw.Draw(tile)

            img_path = data_root / "images" / item.filename
            lbl_path = data_root / "labels" / (Path(item.filename).stem + ".txt")
            gt_coords = _read_yolo_polygon(lbl_path)

            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    w, h = img.size
                    scale = min(tile_size / float(w), tile_size / float(h))
                    new_w = max(1, int(round(float(w) * scale)))
                    new_h = max(1, int(round(float(h) * scale)))
                    img_r = img.resize((new_w, new_h), Image.BILINEAR)
                    offx = (tile_size - new_w) // 2
                    offy = (tile_size - new_h) // 2
                    tile.paste(img_r, (offx, offy))

                    # GT polygon (green)
                    if gt_coords is not None:
                        _draw_quad_on_tile(
                            draw=draw,
                            coords=gt_coords,
                            w=w,
                            h=h,
                            scale=scale,
                            offx=offx,
                            offy=offy,
                            color=(40, 255, 40),
                            width=2,
                        )

                    # Teacher polygon (red) if available for this split
                    teacher_poly = teacher_coords_by_split.get(item.split, {}).get(item.filename, None)
                    if teacher_poly is not None:
                        _draw_quad_on_tile(
                            draw=draw,
                            coords=teacher_poly,
                            w=w,
                            h=h,
                            scale=scale,
                            offx=offx,
                            offy=offy,
                            color=(255, 60, 60),
                            width=2,
                        )
            except Exception:
                pass

            # tiny marker if teacher undetected (top-left)
            if item.teacher_detected == 0:
                draw.rectangle([2, 2, 14, 14], fill=(255, 60, 60))

            collage.paste(tile, (ox, oy))

        out_path = out_dir / "collages" / f"iou_{float(args.iou_min):.2f}_{float(args.iou_max):.2f}_{pidx+1:03d}.{args.collage_format}"
        if str(args.collage_format) == "png":
            collage.save(out_path, optimize=True)
        else:
            collage.save(out_path, quality=int(args.jpeg_quality), optimize=True)

    print(f"\nDone. Collages: {out_dir / 'collages'}")


if __name__ == "__main__":
    main()
