#!/usr/bin/env python3
"""
Create review collages for the Consensus4 dataset build:
  - CHANGED labels: show PRE (backup GT) vs POST (current GT) side-by-side.
  - DROPPED samples: show GT + (optionally) Teacher/Winner/DocAligner overlays.
  - UNCERTAIN kept samples: show GT + overlays for low-IoU single-teacher cases.

This is meant as a quick visual sanity-check to review what changed and what was dropped,
without any web UI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _load_default_font(size: int) -> ImageFont.ImageFont:
    try:
        # Common path on many Linux/macOS installs.
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _read_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("images/"):
            ln = ln[len("images/") :]
        elif ln.startswith("images-negative/"):
            ln = ln[len("images-negative/") :]
        out.append(ln)
    return out


def _load_yolo_polygon(label_path: Path) -> Optional[np.ndarray]:
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
    if coords.shape != (8,) or not np.all(np.isfinite(coords)):
        return None
    return np.clip(coords, 0.0, 1.0)


@dataclass(frozen=True)
class Pred:
    detected: int
    conf: float
    coords: Optional[np.ndarray]  # [8] normalized


def _load_pred_subset(path: Path, wanted: set[str]) -> dict[str, Pred]:
    """
    Load subset of a <split>_*.txt prediction file (11 tokens):
      filename detected conf x0 y0 x1 y1 x2 y2 x3 y3
    """
    out: dict[str, Pred] = {}
    if not path.exists() or not wanted:
        return out
    with path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 11:
                continue
            name = parts[0]
            if name not in wanted:
                continue
            try:
                detected = int(parts[1])
                conf = float(parts[2])
                coords = np.array([float(x) for x in parts[3:11]], dtype=np.float32)
            except Exception:
                continue
            if detected != 1:
                out[name] = Pred(detected=0, conf=conf, coords=None)
                continue
            if coords.shape != (8,) or not np.all(np.isfinite(coords)) or np.any(coords < 0.0):
                out[name] = Pred(detected=0, conf=conf, coords=None)
                continue
            out[name] = Pred(detected=1, conf=conf, coords=np.clip(coords, 0.0, 1.0))
    return out


def _resolve_image_path(data_root: Path, name: str) -> Optional[Path]:
    p = data_root / "images" / name
    if p.exists():
        return p
    p2 = data_root / "images-negative" / name
    if p2.exists():
        return p2
    return None


def _thumb_with_overlays(
    img_path: Path,
    size: int,
    overlays: list[tuple[np.ndarray, tuple[int, int, int], int]],
    title: str = "",
    footer: str = "",
) -> Image.Image:
    """
    overlays: list of (coords8_norm, rgb_color, line_width)
    """
    bg = (18, 18, 18)
    tile = Image.new("RGB", (size, size), bg)

    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w0, h0 = im.size
            scale = min(size / max(1, w0), size / max(1, h0))
            nw = max(1, int(round(w0 * scale)))
            nh = max(1, int(round(h0 * scale)))
            im2 = im.resize((nw, nh), Image.BILINEAR)
            ox = (size - nw) // 2
            oy = (size - nh) // 2
            tile.paste(im2, (ox, oy))

        draw = ImageDraw.Draw(tile)
        for coords, color, lw in overlays:
            if coords is None:
                continue
            coords = np.asarray(coords, dtype=np.float32).reshape(-1)
            if coords.shape != (8,) or not np.all(np.isfinite(coords)):
                continue
            pts: list[tuple[float, float]] = []
            for i in range(0, 8, 2):
                x = float(coords[i]) * float(nw) + float(ox)
                y = float(coords[i + 1]) * float(nh) + float(oy)
                pts.append((x, y))
            pts.append(pts[0])
            draw.line(pts, fill=color, width=int(lw))
            for x, y in pts[:-1]:
                r = 2 + int(max(0, lw - 1))
                draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=1)

        if title:
            font = _load_default_font(12)
            draw.rectangle((0, 0, size, 16), fill=(0, 0, 0))
            draw.text((4, 1), title, fill=(255, 255, 255), font=font)

        if footer:
            font = _load_default_font(12)
            draw.rectangle((0, size - 16, size, size), fill=(0, 0, 0))
            draw.text((4, size - 15), footer, fill=(255, 255, 255), font=font)

    except Exception:
        # placeholder
        draw = ImageDraw.Draw(tile)
        font = _load_default_font(14)
        draw.rectangle((0, 0, size, size), fill=(32, 0, 0))
        draw.text((8, 8), "MISSING/ERROR", fill=(255, 255, 255), font=font)

    return tile


def _make_changed_tile(
    data_root: Path,
    backup_labels: Path,
    labels_dir: Path,
    filename: str,
    panel_size: int,
    meta: dict[str, dict],
) -> Image.Image:
    img_path = _resolve_image_path(data_root, filename)
    if img_path is None:
        img_path = data_root / "images" / filename

    stem = Path(filename).stem
    pre = _load_yolo_polygon(backup_labels / f"{stem}.txt")
    post = _load_yolo_polygon(labels_dir / f"{stem}.txt")

    # PRE in red, POST in green
    pre_img = _thumb_with_overlays(
        img_path=img_path,
        size=panel_size,
        overlays=[(pre, (255, 64, 64), 3)] if pre is not None else [],
        title="PRE",
    )
    post_img = _thumb_with_overlays(
        img_path=img_path,
        size=panel_size,
        overlays=[(post, (64, 255, 64), 3)] if post is not None else [],
        title="POST",
    )

    gap = 4
    w = panel_size * 2 + gap
    h = panel_size + 18
    tile = Image.new("RGB", (w, h), (10, 10, 10))
    tile.paste(pre_img, (0, 0))
    tile.paste(post_img, (panel_size + gap, 0))

    info = meta.get(filename, {})
    action = str(info.get("action", ""))
    chosen = str(info.get("chosen_source", ""))
    footer = f"{chosen}:{action}" if chosen or action else ""

    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, panel_size, w, h), fill=(0, 0, 0))
    font = _load_default_font(12)
    draw.text((4, panel_size + 1), footer[:80], fill=(255, 255, 255), font=font)
    return tile


def _make_dropped_tile(
    data_root: Path,
    labels_dir: Path,
    filename: str,
    tile_size: int,
    preds: dict[str, dict[str, Pred]],
    meta: dict[str, dict],
) -> Image.Image:
    img_path = _resolve_image_path(data_root, filename)
    if img_path is None:
        img_path = data_root / "images" / filename

    stem = Path(filename).stem
    gt = _load_yolo_polygon(labels_dir / f"{stem}.txt")

    overlays: list[tuple[np.ndarray, tuple[int, int, int], int]] = []
    if gt is not None:
        overlays.append((gt, (64, 255, 64), 3))  # GT green

    for key, color in [
        ("teacher", (255, 64, 64)),
        ("winner", (64, 160, 255)),
        ("docaligner", (255, 210, 64)),
    ]:
        p = preds.get(key, {}).get(filename)
        if p is None or p.detected != 1 or p.coords is None:
            continue
        overlays.append((p.coords, color, 2))

    info = meta.get(filename, {})
    reason = str(info.get("reason", "drop"))

    return _thumb_with_overlays(
        img_path=img_path,
        size=tile_size,
        overlays=overlays,
        title="DROPPED",
        footer=reason[:40],
    )


def _make_uncertain_tile(
    data_root: Path,
    labels_dir: Path,
    filename: str,
    tile_size: int,
    preds: dict[str, dict[str, Pred]],
    meta: dict[str, dict],
) -> Image.Image:
    """
    Visualize a kept sample that looks suspicious because GT disagrees with the only available teacher
    (e.g. max_pairwise_iou < 0.50 but we didn't drop because we require >=3 sources to drop).
    """
    img_path = _resolve_image_path(data_root, filename)
    if img_path is None:
        img_path = data_root / "images" / filename

    stem = Path(filename).stem
    gt = _load_yolo_polygon(labels_dir / f"{stem}.txt")

    overlays: list[tuple[np.ndarray, tuple[int, int, int], int]] = []
    if gt is not None:
        overlays.append((gt, (64, 255, 64), 3))  # GT green

    # Draw any available teacher predictions (may be missing in these cases).
    present: list[str] = []
    for key, color in [
        ("teacher", (255, 64, 64)),
        ("winner", (64, 160, 255)),
        ("docaligner", (255, 210, 64)),
    ]:
        p = preds.get(key, {}).get(filename)
        if p is None or p.detected != 1 or p.coords is None:
            continue
        present.append(key)
        overlays.append((p.coords, color, 2))

    info = meta.get(filename, {})
    max_iou = str(info.get("max_pairwise_iou", "")) or "?"

    # Show a compact “why this is suspicious” footer.
    # Example: "avail1(W) iou<0.5 max=0.32"
    short = "".join(
        [
            "T" if "teacher" in present else "",
            "W" if "winner" in present else "",
            "D" if "docaligner" in present else "",
        ]
    )
    short = short or "?"
    footer = f"lowIoU avail1({short}) max={max_iou}"

    return _thumb_with_overlays(
        img_path=img_path,
        size=tile_size,
        overlays=overlays,
        title="UNCERTAIN",
        footer=footer[:40],
    )


def _write_pages(
    out_dir: Path,
    prefix: str,
    tiles_iter,
    tile_w: int,
    tile_h: int,
    cols: int,
    per_page: int,
    header_lines: list[str],
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = max(1, int(cols))
    per_page = max(1, int(per_page))
    rows = int(math.ceil(per_page / cols))
    gap = 4
    margin = 10
    header_h = 22 * max(1, len(header_lines)) + 10

    page_w = margin * 2 + cols * tile_w + (cols - 1) * gap
    page_h = margin * 2 + header_h + rows * tile_h + (rows - 1) * gap

    font = _load_default_font(16)
    saved: list[Path] = []

    page_idx = 1
    page = Image.new("RGB", (page_w, page_h), (8, 8, 8))
    draw = ImageDraw.Draw(page)
    y = margin
    for line in header_lines:
        draw.text((margin, y), line, fill=(255, 255, 255), font=font)
        y += 22

    tile_idx = 0
    placed_total = 0

    def flush() -> None:
        nonlocal page_idx, page, tile_idx, placed_total
        if tile_idx == 0:
            return
        out_path = out_dir / f"{prefix}_page{page_idx:04d}.png"
        page.save(out_path, optimize=True)
        saved.append(out_path)
        page_idx += 1
        tile_idx = 0
        placed_total = 0

    for tile in tiles_iter:
        if tile_idx >= per_page:
            flush()
            # new page
            page = Image.new("RGB", (page_w, page_h), (8, 8, 8))
            draw = ImageDraw.Draw(page)
            y = margin
            for line in header_lines:
                draw.text((margin, y), line, fill=(255, 255, 255), font=font)
                y += 22
            tile_idx = 0
            placed_total = 0

        r = tile_idx // cols
        c = tile_idx % cols
        x0 = margin + c * (tile_w + gap)
        y0 = margin + header_h + r * (tile_h + gap)
        page.paste(tile, (x0, y0))
        tile_idx += 1
        placed_total += 1

    flush()
    return saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create collages for reviewing Consensus4 dataset changes/drops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new",
        help="Dataset root",
    )
    p.add_argument("--mode", type=str, default="both", choices=["changed", "dropped", "uncertain", "both"])
    p.add_argument("--out_dir", type=str, default="consensus4_review_collages", help="Output directory (under data_root)")
    p.add_argument("--panel_size", type=int, default=160, help="Per-panel size for CHANGED tiles (PRE/POST)")
    p.add_argument("--dropped_size", type=int, default=200, help="Tile size for DROPPED tiles")
    p.add_argument(
        "--uncertain_csv",
        type=str,
        default="",
        help="CSV list for 'uncertain' mode (default: data_root/consensus4_kept_max_pairwise_iou_lt_0.50_avail1.csv)",
    )
    p.add_argument("--cols", type=int, default=20, help="Columns per page")
    p.add_argument("--per_page", type=int, default=400, help="Tiles per page")
    p.add_argument("--max_items", type=int, default=0, help="If >0, limit items for each mode")
    p.add_argument("--backup_dir", type=str, default="", help="labels_backup_consensus4_* dir (default: from summary)")
    p.add_argument("--decisions_csv", type=str, default="", help="consensus4_decisions.csv path (default: data_root)")
    p.add_argument("--changed_list", type=str, default="", help="consensus4_labels_changed.txt path (default: data_root)")
    p.add_argument("--dropped_list", type=str, default="", help="consensus4_dropped.txt path (default: data_root)")
    p.add_argument("--teacher_suffix", type=str, default="teacher", help="Suffix for <split>_<suffix>.txt (teacher preds)")
    p.add_argument("--winner_suffix", type=str, default="winner", help="Suffix for <split>_<suffix>.txt (winner preds)")
    p.add_argument("--docaligner_suffix", type=str, default="docaligner", help="Suffix for <split>_<suffix>.txt (docaligner preds)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = data_root / str(args.out_dir)
    labels_dir = data_root / "labels"

    summary_path = data_root / "consensus4_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    backup_dir = Path(args.backup_dir) if args.backup_dir else Path(summary.get("backup_dir", ""))
    if args.mode in {"changed", "both"} and (not backup_dir or not backup_dir.exists()):
        raise SystemExit(
            "Missing backup_dir. Provide --backup_dir or ensure consensus4_summary.json contains backup_dir."
        )

    decisions_path = Path(args.decisions_csv) if args.decisions_csv else (data_root / "consensus4_decisions.csv")
    if not decisions_path.exists():
        raise SystemExit(f"Missing decisions CSV: {decisions_path}")

    changed_list = Path(args.changed_list) if args.changed_list else (data_root / "consensus4_labels_changed.txt")
    dropped_list = Path(args.dropped_list) if args.dropped_list else (data_root / "consensus4_dropped.txt")

    meta: dict[str, dict] = {}
    with decisions_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            meta[row["filename"]] = row

    saved_paths: list[Path] = []

    if args.mode in {"changed", "both"}:
        changed = _read_list(changed_list)
        if args.max_items and args.max_items > 0:
            changed = changed[: int(args.max_items)]

        def tiles():
            for fn in tqdm(changed, desc="Render CHANGED tiles"):
                yield _make_changed_tile(
                    data_root=data_root,
                    backup_labels=backup_dir,
                    labels_dir=labels_dir,
                    filename=fn,
                    panel_size=int(args.panel_size),
                    meta=meta,
                )

        tile_w = int(args.panel_size) * 2 + 4
        tile_h = int(args.panel_size) + 18
        hdr = [
            "Consensus4 CHANGED labels",
            "PRE=backup GT (red)  POST=current GT (green)",
        ]
        saved_paths.extend(
            _write_pages(
                out_dir=out_dir / "changed",
                prefix="changed",
                tiles_iter=tiles(),
                tile_w=tile_w,
                tile_h=tile_h,
                cols=int(args.cols),
                per_page=int(args.per_page),
                header_lines=hdr,
            )
        )

    if args.mode in {"dropped", "both"}:
        dropped = _read_list(dropped_list)
        if args.max_items and args.max_items > 0:
            dropped = dropped[: int(args.max_items)]

        wanted = set(dropped)
        # Load only needed predictions for dropped items (from both splits).
        preds: dict[str, dict[str, Pred]] = {"teacher": {}, "winner": {}, "docaligner": {}}
        for split in ["train", "val"]:
            for key, suffix in [
                ("teacher", str(args.teacher_suffix)),
                ("winner", str(args.winner_suffix)),
                ("docaligner", str(args.docaligner_suffix)),
            ]:
                p = data_root / f"{split}_{suffix}.txt"
                preds[key].update(_load_pred_subset(p, wanted))

        def tiles():
            for fn in tqdm(dropped, desc="Render DROPPED tiles"):
                yield _make_dropped_tile(
                    data_root=data_root,
                    labels_dir=labels_dir,
                    filename=fn,
                    tile_size=int(args.dropped_size),
                    preds=preds,
                    meta=meta,
                )

        tile_w = int(args.dropped_size)
        tile_h = int(args.dropped_size)
        hdr = [
            "Consensus4 DROPPED samples",
            "GT=green  Teacher=red  Winner=blue  DocAligner=yellow",
        ]
        saved_paths.extend(
            _write_pages(
                out_dir=out_dir / "dropped",
                prefix="dropped",
                tiles_iter=tiles(),
                tile_w=tile_w,
                tile_h=tile_h,
                cols=int(args.cols),
                per_page=int(args.per_page),
                header_lines=hdr,
            )
        )

    if args.mode == "uncertain":
        uncertain_csv = Path(args.uncertain_csv) if args.uncertain_csv else (
            data_root / "consensus4_kept_max_pairwise_iou_lt_0.50_avail1.csv"
        )
        if not uncertain_csv.exists():
            raise SystemExit(f"Missing uncertain CSV: {uncertain_csv}")

        wanted: list[str] = []
        with uncertain_csv.open(newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                fn = (row.get("filename") or "").strip()
                if fn:
                    wanted.append(fn)
        if args.max_items and args.max_items > 0:
            wanted = wanted[: int(args.max_items)]

        wanted_set = set(wanted)
        preds: dict[str, dict[str, Pred]] = {"teacher": {}, "winner": {}, "docaligner": {}}
        for split in ["train", "val"]:
            for key, suffix in [
                ("teacher", str(args.teacher_suffix)),
                ("winner", str(args.winner_suffix)),
                ("docaligner", str(args.docaligner_suffix)),
            ]:
                p = data_root / f"{split}_{suffix}.txt"
                preds[key].update(_load_pred_subset(p, wanted_set))

        def tiles():
            for fn in tqdm(wanted, desc="Render UNCERTAIN tiles"):
                yield _make_uncertain_tile(
                    data_root=data_root,
                    labels_dir=labels_dir,
                    filename=fn,
                    tile_size=int(args.dropped_size),
                    preds=preds,
                    meta=meta,
                )

        hdr = [
            "Consensus4 UNCERTAIN kept samples",
            "Filter: max_pairwise_iou < 0.50 AND only 1 teacher available",
            "GT=green  Teacher=red  Winner=blue  DocAligner=yellow",
        ]
        saved_paths.extend(
            _write_pages(
                out_dir=out_dir / "uncertain_avail1_iou_lt_0.50",
                prefix="uncertain",
                tiles_iter=tiles(),
                tile_w=int(args.dropped_size),
                tile_h=int(args.dropped_size),
                cols=int(args.cols),
                per_page=int(args.per_page),
                header_lines=hdr,
            )
        )

    # Write a small manifest
    manifest = {
        "data_root": str(data_root),
        "generated_at": _timestamp(),
        "out_dir": str(out_dir),
        "mode": str(args.mode),
        "changed_count": len(_read_list(changed_list)) if changed_list.exists() else 0,
        "dropped_count": len(_read_list(dropped_list)) if dropped_list.exists() else 0,
        "saved": [str(p) for p in saved_paths],
        "legend": {
            "changed": {"pre": "red", "post": "green"},
            "dropped": {"gt": "green", "teacher": "red", "winner": "blue", "docaligner": "yellow"},
            "uncertain": {"gt": "green", "teacher": "red", "winner": "blue", "docaligner": "yellow"},
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\nDone.")
    print(f"Output dir: {out_dir}")
    print(f"Pages: {len(saved_paths)} (see manifest.json)")


if __name__ == "__main__":
    main()
