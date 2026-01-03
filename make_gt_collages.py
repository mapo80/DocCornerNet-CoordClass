#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: Pillow (pip install pillow)") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _read_split(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _load_label(label_path: Path) -> np.ndarray | None:
    if not label_path.exists():
        return None
    line = label_path.read_text(encoding="utf-8").strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 9:
        return None
    try:
        coords = np.array([float(x) for x in parts[1:9]], dtype=np.float32)
    except Exception:
        return None
    return coords


def _draw_quad(draw: ImageDraw.ImageDraw, coords_norm8: np.ndarray, size: int) -> None:
    pts = [(float(coords_norm8[i]) * size, float(coords_norm8[i + 1]) * size) for i in range(0, 8, 2)]
    pts.append(pts[0])
    draw.line(pts, fill=(0, 255, 0), width=max(1, size // 64))


def _progress(it, total: int, desc: str):
    if tqdm is None:
        return it
    return tqdm(it, total=total, desc=desc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create GT collages (500 thumbnails per image).")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(
            "/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-last"
        ),
    )
    parser.add_argument("--split", type=str, choices=["train", "val", "both"], default="both")
    parser.add_argument(
        "--prefixes",
        type=str,
        default="",
        help="Optional comma-separated filename prefixes to include (case-insensitive).",
    )
    parser.add_argument("--per_collage", type=int, default=500)
    parser.add_argument("--cols", type=int, default=20)
    parser.add_argument("--tile", type=int, default=96, help="Tile size in pixels (square).")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to <data_root>/gt_collages_<per_collage>.",
    )
    parser.add_argument("--jpeg_quality", type=int, default=90)
    args = parser.parse_args()

    if args.per_collage <= 0:
        raise SystemExit("--per_collage must be > 0")
    if args.cols <= 0:
        raise SystemExit("--cols must be > 0")
    if args.tile <= 16:
        raise SystemExit("--tile too small")

    rows = int(math.ceil(args.per_collage / args.cols))
    prefixes = [p.strip().lower() for p in str(args.prefixes).split(",") if p.strip()]
    data_root: Path = args.data_root
    if args.output_dir is None:
        suffix = ""
        if prefixes:
            suffix = "_" + "_".join(prefixes)
        out_root = data_root / f"gt_collages_{args.per_collage}{suffix}"
    else:
        out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    images_dir = data_root / "images"
    labels_dir = data_root / "labels"
    neg_images_dir = data_root / "images-negative"
    neg_labels_dir = data_root / "labels-negative"

    for p in [images_dir, labels_dir, neg_images_dir, neg_labels_dir]:
        if not p.exists():
            raise SystemExit(f"Missing: {p}")

    def make_for_split(split: str) -> None:
        split_path = data_root / f"{split}.txt"
        if not split_path.exists():
            raise SystemExit(f"Missing split file: {split_path}")

        names = _read_split(split_path)
        if prefixes:
            names = [n for n in names if any(n.lower().startswith(p) for p in prefixes)]
        n = len(names)
        if n == 0:
            print(f"No images matched in {split} (prefixes={prefixes or ['<all>']})")
            return
        pages = int(math.ceil(n / args.per_collage))
        split_out = out_root / split
        split_out.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(total=n, desc=f"Building collages ({split})") if tqdm else None
        for page in range(pages):
            start = page * args.per_collage
            end = min(n, (page + 1) * args.per_collage)
            chunk = names[start:end]

            collage = Image.new(
                "RGB",
                (args.cols * args.tile, rows * args.tile),
                (16, 16, 16),
            )

            for i, name in enumerate(chunk):
                col = i % args.cols
                row = i // args.cols
                x0 = col * args.tile
                y0 = row * args.tile

                is_neg = name.lower().startswith("negative_")
                img_path = (neg_images_dir / name) if is_neg else (images_dir / name)
                lbl_path = (
                    (neg_labels_dir / f"{Path(name).stem}.txt")
                    if is_neg
                    else (labels_dir / f"{Path(name).stem}.txt")
                )

                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize((args.tile, args.tile), Image.BILINEAR)
                except Exception:
                    img = Image.new("RGB", (args.tile, args.tile), (80, 80, 80))

                coords = _load_label(lbl_path)
                if coords is not None:
                    draw = ImageDraw.Draw(img)
                    _draw_quad(draw, coords, args.tile)

                collage.paste(img, (x0, y0))
                if pbar:
                    pbar.update(1)

            out_path = split_out / f"{split}_{page:04d}.jpg"
            collage.save(out_path, quality=int(args.jpeg_quality), optimize=True)

        if pbar:
            pbar.close()
        print(f"Saved {pages} collage(s) to {split_out}")

    if args.split in {"train", "both"}:
        make_for_split("train")
    if args.split in {"val", "both"}:
        make_for_split("val")


if __name__ == "__main__":
    main()
