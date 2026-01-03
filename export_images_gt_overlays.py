#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

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


def _parse_prefixes(prefixes_csv: str) -> list[str]:
    return [p.strip().lower() for p in str(prefixes_csv).split(",") if p.strip()]


def _load_polys_norm(label_path: Path) -> list[list[float]]:
    if not label_path.exists():
        return []
    polys: list[list[float]] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        try:
            coords = [float(x) for x in parts[1:9]]
        except Exception:
            continue
        polys.append(coords)
    return polys


def _draw_poly(draw: ImageDraw.ImageDraw, coords_norm8: list[float], w: int, h: int) -> None:
    pts = [(coords_norm8[i] * w, coords_norm8[i + 1] * h) for i in range(0, 8, 2)]
    pts2 = pts + [pts[0]]
    stroke = max(2, int(round(max(w, h) / 200)))
    draw.line(pts2, fill=(0, 255, 0), width=stroke, joint="curve")

    r = max(2, int(round(max(w, h) / 180)))
    for x, y in pts:
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0), width=max(1, r // 2))


@dataclass(frozen=True)
class _Item:
    split: str
    name: str
    is_negative: bool
    image_path: Path
    label_path: Path
    overlay_rel: str


def _iter_items(
    data_root: Path,
    split: str,
    prefixes: list[str],
    include_negatives: bool,
    output_dir: Path,
) -> list[_Item]:
    split_path = data_root / f"{split}.txt"
    if not split_path.exists():
        raise SystemExit(f"Missing split file: {split_path}")

    names = _read_split(split_path)
    if prefixes:
        names = [n for n in names if any(n.lower().startswith(p) for p in prefixes)]

    items: list[_Item] = []
    images_dir = data_root / "images"
    labels_dir = data_root / "labels"
    neg_images_dir = data_root / "images-negative"
    neg_labels_dir = data_root / "labels-negative"

    for name in names:
        is_negative = name.lower().startswith("negative_")
        if is_negative and not include_negatives:
            continue
        img_path = (neg_images_dir / name) if is_negative else (images_dir / name)
        lbl_path = (
            (neg_labels_dir / f"{Path(name).stem}.txt")
            if is_negative
            else (labels_dir / f"{Path(name).stem}.txt")
        )
        overlay_rel = str(Path(split) / name)
        items.append(
            _Item(
                split=split,
                name=name,
                is_negative=is_negative,
                image_path=img_path,
                label_path=lbl_path,
                overlay_rel=overlay_rel,
            )
        )
    return items


def _save_image(img: Image.Image, out_path: Path, jpeg_quality: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        img.save(out_path, quality=int(jpeg_quality), optimize=True)
    else:
        img.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per-image overlays (image + GT polygon) for manual review (delete bad ones)."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(
            "/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-last"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to <data_root>/images-gt.",
    )
    parser.add_argument("--split", type=str, choices=["train", "val", "both"], default="both")
    parser.add_argument(
        "--prefixes",
        type=str,
        default="smartdoc",
        help="Comma-separated filename prefixes to include (case-insensitive). Default: smartdoc",
    )
    parser.add_argument(
        "--include_negatives",
        action="store_true",
        help="Also export negatives (usually not needed for bbox review).",
    )
    parser.add_argument(
        "--max_side",
        type=int,
        default=640,
        help="Resize keeping aspect so max(width,height)=max_side. Set 0 to keep original.",
    )
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing overlays.")
    args = parser.parse_args()

    data_root: Path = args.data_root
    if not data_root.exists():
        raise SystemExit(f"Missing dataset root: {data_root}")

    required = [
        data_root / "images",
        data_root / "labels",
        data_root / "images-negative",
        data_root / "labels-negative",
        data_root / "train.txt",
        data_root / "val.txt",
    ]
    for p in required:
        if not p.exists():
            raise SystemExit(f"Missing: {p}")

    prefixes = _parse_prefixes(args.prefixes)
    out_root = args.output_dir if args.output_dir is not None else (data_root / "images-gt")
    out_root.mkdir(parents=True, exist_ok=True)

    index_path = out_root / "index.jsonl"
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_meta_path = out_root / f"run_{ts}.json"

    items: list[_Item] = []
    if args.split in {"train", "both"}:
        items.extend(
            _iter_items(
                data_root=data_root,
                split="train",
                prefixes=prefixes,
                include_negatives=bool(args.include_negatives),
                output_dir=out_root,
            )
        )
    if args.split in {"val", "both"}:
        items.extend(
            _iter_items(
                data_root=data_root,
                split="val",
                prefixes=prefixes,
                include_negatives=bool(args.include_negatives),
                output_dir=out_root,
            )
        )

    if not items:
        raise SystemExit(f"No items matched (split={args.split}, prefixes={prefixes or ['<all>']}).")

    total = len(items)
    pbar = tqdm(total=total, desc="Exporting overlays") if tqdm else None

    written = 0
    missing = 0
    with index_path.open("w", encoding="utf-8") as idx:
        for item in items:
            out_path = out_root / item.overlay_rel
            if out_path.exists() and not args.overwrite:
                rec = {
                    "split": item.split,
                    "name": item.name,
                    "overlay_rel": item.overlay_rel,
                    "is_negative": item.is_negative,
                    "image_rel": str(item.image_path.relative_to(data_root)),
                    "label_rel": str(item.label_path.relative_to(data_root)),
                    "skipped_existing": True,
                }
                idx.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if pbar:
                    pbar.update(1)
                continue

            if not item.image_path.exists():
                rec = {
                    "split": item.split,
                    "name": item.name,
                    "overlay_rel": item.overlay_rel,
                    "is_negative": item.is_negative,
                    "image_rel": str(item.image_path.relative_to(data_root)),
                    "label_rel": str(item.label_path.relative_to(data_root)),
                    "error": "missing_image",
                }
                idx.write(json.dumps(rec, ensure_ascii=False) + "\n")
                missing += 1
                if pbar:
                    pbar.update(1)
                continue

            try:
                img = Image.open(item.image_path).convert("RGB")
            except Exception:
                rec = {
                    "split": item.split,
                    "name": item.name,
                    "overlay_rel": item.overlay_rel,
                    "is_negative": item.is_negative,
                    "image_rel": str(item.image_path.relative_to(data_root)),
                    "label_rel": str(item.label_path.relative_to(data_root)),
                    "error": "failed_open_image",
                }
                idx.write(json.dumps(rec, ensure_ascii=False) + "\n")
                missing += 1
                if pbar:
                    pbar.update(1)
                continue

            if args.max_side and args.max_side > 0:
                w0, h0 = img.size
                m0 = max(w0, h0)
                if m0 > args.max_side:
                    scale = float(args.max_side) / float(m0)
                    w1 = max(1, int(round(w0 * scale)))
                    h1 = max(1, int(round(h0 * scale)))
                    img = img.resize((w1, h1), Image.BILINEAR)

            polys = _load_polys_norm(item.label_path)
            if polys:
                draw = ImageDraw.Draw(img)
                w, h = img.size
                for coords in polys:
                    _draw_poly(draw, coords, w=w, h=h)

            _save_image(img, out_path, jpeg_quality=int(args.jpeg_quality))
            rec = {
                "split": item.split,
                "name": item.name,
                "overlay_rel": item.overlay_rel,
                "is_negative": item.is_negative,
                "image_rel": str(item.image_path.relative_to(data_root)),
                "label_rel": str(item.label_path.relative_to(data_root)),
                "written": True,
            }
            idx.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()

    meta = {
        "data_root": str(data_root),
        "output_dir": str(out_root),
        "index": str(index_path),
        "split": args.split,
        "prefixes": prefixes,
        "include_negatives": bool(args.include_negatives),
        "max_side": int(args.max_side),
        "jpeg_quality": int(args.jpeg_quality),
        "overwrite": bool(args.overwrite),
        "total_items": total,
        "written": written,
        "missing_or_failed": missing,
    }
    run_meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Output: {out_root}")
    print(f"Index:  {index_path}")
    print(f"Meta:   {run_meta_path}")
    print(f"Done: written={written} missing/failed={missing} total={total}")


if __name__ == "__main__":
    main()

