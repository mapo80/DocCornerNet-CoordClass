#!/usr/bin/env python3
"""
Identify "undesirable" GT samples (for review), confirmed with a teacher.

We target (positives only):
  - Near-border documents (polygon bbox too close to image border)
  - Small documents (polygon area/bbox dims too small)
  - Weird geometries (self-intersections, non-convex, extreme angles/edge ratios, low fill ratio)

Confirmation with the teacher is done by joining against an existing audit CSV:
  teacher_vs_gt_full.csv (produced by teacher_clean_dataset.py)

Outputs:
  - candidates.csv: GT-flagged samples + metrics + teacher stats
  - exclude_candidates.csv: subset confirmed by teacher (conf+IoU thresholds)
  - exclude_candidates_{train,val}.txt: candidates to review/exclude (NOT applied automatically)
  - collages: thumbnails (1500/page) of exclude candidates, with GT (green) + teacher (red)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFile
from tqdm import tqdm

try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid

    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False


ImageFile.LOAD_TRUNCATED_IMAGES = True


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _read_split_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text().splitlines():
        name = raw.strip()
        if not name:
            continue
        if name.startswith("images/"):
            name = name[len("images/") :]
        elif name.startswith("images-negative/"):
            name = name[len("images-negative/") :]
        lines.append(name)
    return lines


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.write_text("".join(f"{l}\n" for l in lines))


def _load_yolo_polygon(label_path: Path) -> Optional[np.ndarray]:
    """
    Returns normalized coords [8] or None if missing/invalid.
    YOLO polygon format: class x0 y0 x1 y1 x2 y2 x3 y3
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


def _polygon_area_ratio(pts_norm: np.ndarray) -> float:
    """
    pts_norm: [4,2] in normalized coordinates. Returns area in [0..1] approx.
    """
    if pts_norm.shape != (4, 2):
        return 0.0

    if _HAS_SHAPELY:
        try:
            poly = Polygon([(float(x), float(y)) for x, y in pts_norm])
            if not poly.is_valid:
                poly = make_valid(poly)
                if poly.geom_type == "GeometryCollection":
                    polys = [g for g in poly.geoms if g.geom_type == "Polygon"]
                    poly = max(polys, key=lambda p: p.area) if polys else Polygon(pts_norm).convex_hull
                elif poly.geom_type == "MultiPolygon":
                    poly = max(poly.geoms, key=lambda p: p.area)
            return float(poly.area) if not poly.is_empty else 0.0
        except Exception:
            return 0.0

    # Shoelace (on given order)
    x = pts_norm[:, 0]
    y = pts_norm[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _order_points_ccw(pts: np.ndarray) -> np.ndarray:
    """
    Order points counter-clockwise by angle around centroid.
    This is robust for convex quads and also "repairs" many order issues.
    """
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(angles)
    return pts[order]


def _edge_lengths(pts: np.ndarray) -> np.ndarray:
    diffs = np.roll(pts, -1, axis=0) - pts
    return np.linalg.norm(diffs, axis=1)


def _polygon_angles_deg(pts: np.ndarray) -> np.ndarray:
    """
    pts: [N,2] ordered CCW.
    Returns internal angles in degrees for each vertex.
    """
    n = pts.shape[0]
    angles = []
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p = pts[i]
        p_next = pts[(i + 1) % n]
        v1 = p_prev - p
        v2 = p_next - p
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 <= 1e-9 or n2 <= 1e-9:
            angles.append(0.0)
            continue
        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        ang = math.degrees(math.acos(cosang))
        angles.append(ang)
    return np.array(angles, dtype=np.float32)


def _is_convex_ccw(pts: np.ndarray) -> bool:
    """
    pts: [N,2] ordered CCW. Returns True if all cross products have same sign.
    """
    n = pts.shape[0]
    if n < 4:
        return True
    signs = []
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        c = pts[(i + 2) % n]
        ab = b - a
        bc = c - b
        cross = float(ab[0] * bc[1] - ab[1] * bc[0])
        if abs(cross) > 1e-12:
            signs.append(cross)
    if not signs:
        return True
    return (min(signs) > 0) or (max(signs) < 0)


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """Proper segment intersection test (excluding collinear overlaps)."""

    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    # General case
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True

    return False


def _is_self_intersecting_raw(pts: np.ndarray) -> bool:
    """
    pts: [4,2] in the raw GT order.
    For a quad, self-intersection happens if (0-1) intersects (2-3) or (1-2) intersects (3-0).
    """
    if pts.shape != (4, 2):
        return False
    a, b, c, d = pts
    return _segments_intersect(a, b, c, d) or _segments_intersect(b, c, d, a)


@dataclass(frozen=True)
class TeacherRow:
    detected: bool
    conf: float
    iou: Optional[float]
    err_mean_px: Optional[float]
    err_max_px: Optional[float]
    w: Optional[int]
    h: Optional[int]


def _load_teacher_csv(path: Path) -> dict[tuple[str, str], TeacherRow]:
    m: dict[tuple[str, str], TeacherRow] = {}
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            split = str(row.get("split", "")).strip()
            filename = str(row.get("filename", "")).strip()
            if not split or not filename:
                continue
            detected = str(row.get("teacher_detected", "0")).strip() in ("1", "True", "true")

            def fnum(key: str) -> Optional[float]:
                v = str(row.get(key, "")).strip()
                if v == "":
                    return None
                try:
                    return float(v)
                except Exception:
                    return None

            def inum(key: str) -> Optional[int]:
                v = str(row.get(key, "")).strip()
                if v == "":
                    return None
                try:
                    return int(float(v))
                except Exception:
                    return None

            m[(split, filename)] = TeacherRow(
                detected=detected,
                conf=float(fnum("teacher_conf") or 0.0),
                iou=fnum("iou"),
                err_mean_px=fnum("err_mean_px"),
                err_max_px=fnum("err_max_px"),
                w=inum("w"),
                h=inum("h"),
            )
    return m


def _load_teacher_split_file(path: Path) -> dict[str, np.ndarray]:
    """
    Load <split>_teacher.txt created by generate_teacher_gt_splits.py.

    Format (11 tokens):
      filename detected conf x0 y0 x1 y1 x2 y2 x3 y3

    Returns: filename -> coords_norm[8] for detected==1.
    """
    out: dict[str, np.ndarray] = {}
    with path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 11:
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
            if coords.shape != (8,) or np.any(coords < 0.0):
                continue
            out[name] = coords
    return out


@dataclass
class Candidate:
    split: str
    filename: str
    coords: np.ndarray  # [8] normalized
    w: Optional[int]
    h: Optional[int]
    # GT metrics
    border_margin_norm: float
    border_margin_px: Optional[float]
    area_ratio: float
    bbox_area_ratio: float
    bbox_w_norm: float
    bbox_h_norm: float
    min_dim_norm: float
    fill_ratio: float
    self_intersect_raw: bool
    convex: bool
    min_angle: float
    max_angle: float
    edge_ratio: float
    # Flags
    near_border: bool
    small: bool
    weird: bool
    flags: str
    # Teacher
    teacher_detected: bool
    teacher_conf: float
    teacher_iou: Optional[float]
    teacher_err_mean_px: Optional[float]
    teacher_err_max_px: Optional[float]
    teacher_status: str
    teacher_match: bool
    teacher_confirmed: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find GT outliers (border/small/weird) and confirm with teacher audit CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new",
        help="Dataset root (contains images/, labels/, train.txt, val.txt, ...).",
    )
    p.add_argument("--splits", type=str, default="train,val", help="Comma-separated split basenames")
    p.add_argument(
        "--teacher_csv",
        type=str,
        default="",
        help="Path to teacher_vs_gt_full.csv (if empty, defaults to <data_root>/teacher_vs_gt_full.csv).",
    )
    p.add_argument("--confirm_conf", type=float, default=0.90, help="Teacher confidence threshold for confirmation")
    p.add_argument("--confirm_iou", type=float, default=0.90, help="IoU threshold for confirmation")

    # Near-border thresholds
    p.add_argument("--border_norm", type=float, default=0.005, help="Min bbox margin to border (normalized)")
    p.add_argument("--border_px", type=float, default=3.0, help="Min bbox margin to border (pixels)")

    # Small thresholds
    p.add_argument("--small_area", type=float, default=0.05, help="Polygon area ratio threshold (0-1)")
    p.add_argument("--small_min_dim", type=float, default=0.15, help="Min(bbox_w,bbox_h) threshold (0-1)")

    # Weird geometry thresholds
    p.add_argument("--weird_fill_ratio", type=float, default=0.55, help="area / bbox_area below => weird")
    p.add_argument("--weird_edge_ratio", type=float, default=10.0, help="max_edge/min_edge above => weird")
    p.add_argument("--weird_min_angle", type=float, default=20.0, help="min internal angle below => weird")
    p.add_argument("--weird_max_angle", type=float, default=160.0, help="max internal angle above => weird")

    # Output / collage
    p.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory (default: evaluation_results/gt_outliers_teacher_<timestamp>).",
    )
    p.add_argument("--make_collage", action="store_true", help="Generate collage(s) from confirmed samples")
    p.add_argument("--tiles_per_page", type=int, default=1500, help="Number of thumbnails per collage page")
    p.add_argument("--cols", type=int, default=50, help="Columns per collage page")
    p.add_argument("--tile_size", type=int, default=128, help="Thumbnail tile size (square)")
    p.add_argument("--collage_format", type=str, default="jpg", choices=["jpg", "png"], help="Collage image format")
    p.add_argument("--jpeg_quality", type=int, default=85, help="JPEG quality when collage_format=jpg")
    p.add_argument("--max_images", type=int, default=0, help="If >0, limit processed positives (debug)")
    return p.parse_args()


def _draw_quad_on_tile(draw: ImageDraw.ImageDraw, coords: np.ndarray, w: int, h: int, scale: float, offx: int, offy: int, color: tuple[int, int, int], width: int) -> None:
    pts = coords.reshape(4, 2)
    xy = []
    for x, y in pts:
        px = float(x) * float(w) * scale + float(offx)
        py = float(y) * float(h) * scale + float(offy)
        xy.append((px, py))
    xy.append(xy[0])
    draw.line(xy, fill=color, width=width, joint="curve")


def _tile_reason_markers(draw: ImageDraw.ImageDraw, candidate: Candidate) -> None:
    # Small color squares on top-left
    x0, y0 = 2, 2
    s = 10
    if candidate.near_border:
        draw.rectangle([x0, y0, x0 + s, y0 + s], outline=None, fill=(255, 60, 60))
        x0 += s + 2
    if candidate.small:
        draw.rectangle([x0, y0, x0 + s, y0 + s], outline=None, fill=(80, 160, 255))
        x0 += s + 2
    if candidate.weird:
        draw.rectangle([x0, y0, x0 + s, y0 + s], outline=None, fill=(255, 220, 80))


def _make_collages(
    candidates: list[Candidate],
    data_root: Path,
    out_dir: Path,
    tiles_per_page: int,
    cols: int,
    tile_size: int,
    collage_format: str,
    jpeg_quality: int,
    teacher_coords_by_split: dict[str, dict[str, np.ndarray]],
) -> list[Path]:
    if not candidates:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    collages_dir = out_dir / "collages"
    collages_dir.mkdir(parents=True, exist_ok=True)

    rows = int(math.ceil(float(tiles_per_page) / float(cols)))
    page_w = cols * tile_size
    page_h = rows * tile_size

    pages = int(math.ceil(len(candidates) / float(tiles_per_page)))
    out_paths: list[Path] = []

    for pidx in tqdm(range(pages), desc="Collage pages"):
        start = pidx * tiles_per_page
        end = min(len(candidates), start + tiles_per_page)
        page_items = candidates[start:end]

        collage = Image.new("RGB", (page_w, page_h), (18, 18, 18))

        for i, cand in enumerate(page_items):
            col = i % cols
            row = i // cols
            ox = col * tile_size
            oy = row * tile_size

            tile = Image.new("RGB", (tile_size, tile_size), (18, 18, 18))
            draw = ImageDraw.Draw(tile)

            img_path = data_root / "images" / cand.filename
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

                    # GT polygon overlay
                    _draw_quad_on_tile(
                        draw=draw,
                        coords=cand.coords,
                        w=w,
                        h=h,
                        scale=scale,
                        offx=offx,
                        offy=offy,
                        color=(40, 255, 40),
                        width=2,
                    )

                    # Teacher polygon overlay (red), if available
                    teacher_map = teacher_coords_by_split.get(cand.split, {})
                    teacher_coords = teacher_map.get(cand.filename, None)
                    if teacher_coords is not None:
                        _draw_quad_on_tile(
                            draw=draw,
                            coords=teacher_coords,
                            w=w,
                            h=h,
                            scale=scale,
                            offx=offx,
                            offy=offy,
                            color=(255, 60, 60),
                            width=2,
                        )
            except Exception:
                # keep blank tile
                pass

            _tile_reason_markers(draw, cand)
            collage.paste(tile, (ox, oy))

        out_path = collages_dir / f"outliers_confirmed_{pidx+1:03d}.{collage_format}"
        if collage_format == "png":
            collage.save(out_path, optimize=True)
        else:
            collage.save(out_path, quality=int(jpeg_quality), optimize=True)
        out_paths.append(out_path)

    return out_paths


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]

    teacher_csv = Path(args.teacher_csv) if args.teacher_csv else (data_root / "teacher_vs_gt_full.csv")
    if not teacher_csv.exists():
        raise SystemExit(f"Missing teacher audit CSV: {teacher_csv}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("evaluation_results") / f"gt_outliers_teacher_{_timestamp()}"

    out_dir.mkdir(parents=True, exist_ok=True)

    teacher_map = _load_teacher_csv(teacher_csv)

    # Build list of positives in splits
    positives: list[tuple[str, str]] = []
    for split in splits:
        split_file = data_root / f"{split}.txt"
        if not split_file.exists():
            raise SystemExit(f"Missing split file: {split_file}")
        for name in _read_split_lines(split_file):
            if name.startswith("negative_"):
                continue
            positives.append((split, name))

    if args.max_images and args.max_images > 0:
        positives = positives[: int(args.max_images)]

    print("\n" + "=" * 80)
    print("GT outliers (border/small/weird) + teacher confirmation")
    print("=" * 80)
    print(f"Dataset: {data_root}")
    print(f"Teacher CSV: {teacher_csv}")
    print(f"Splits: {splits}")
    print(f"Positives: {len(positives)}")
    print(f"Output: {out_dir}")

    # Scan
    candidates: list[Candidate] = []
    counts = {"near_border": 0, "small": 0, "weird": 0, "any": 0, "confirmed": 0}
    sb_status_counts: dict[str, int] = {}

    for split, filename in tqdm(positives, desc="Scanning GT"):
        lbl_path = data_root / "labels" / (Path(filename).stem + ".txt")
        coords = _load_yolo_polygon(lbl_path)
        if coords is None:
            continue

        pts = coords.reshape(4, 2)
        xs = pts[:, 0]
        ys = pts[:, 1]
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
        bbox_w = max(0.0, xmax - xmin)
        bbox_h = max(0.0, ymax - ymin)
        bbox_area = bbox_w * bbox_h

        border_margin_norm = min(xmin, ymin, 1.0 - xmax, 1.0 - ymax)

        sb = teacher_map.get((split, filename))
        w = sb.w if sb is not None else None
        h = sb.h if sb is not None else None
        border_margin_px = None
        if w is not None and h is not None:
            border_margin_px = min(xmin * w, ymin * h, (1.0 - xmax) * w, (1.0 - ymax) * h)

        pts_ord = _order_points_ccw(pts)
        area_ratio = _polygon_area_ratio(pts_ord)
        fill_ratio = float(area_ratio / bbox_area) if bbox_area > 1e-9 else 0.0

        self_intersect_raw = _is_self_intersecting_raw(pts)
        convex = _is_convex_ccw(pts_ord)
        edges = _edge_lengths(pts_ord)
        min_edge = float(edges.min()) if edges.size else 0.0
        max_edge = float(edges.max()) if edges.size else 0.0
        edge_ratio = float(max_edge / min_edge) if min_edge > 1e-9 else float("inf")

        angles = _polygon_angles_deg(pts_ord)
        min_angle = float(angles.min()) if angles.size else 0.0
        max_angle = float(angles.max()) if angles.size else 0.0

        min_dim_norm = float(min(bbox_w, bbox_h))

        near_border = (border_margin_norm <= float(args.border_norm)) or (
            border_margin_px is not None and border_margin_px <= float(args.border_px)
        )
        small = (area_ratio <= float(args.small_area)) or (min_dim_norm <= float(args.small_min_dim))
        weird = (
            self_intersect_raw
            or (not convex)
            or (fill_ratio <= float(args.weird_fill_ratio))
            or (edge_ratio >= float(args.weird_edge_ratio))
            or (min_angle <= float(args.weird_min_angle))
            or (max_angle >= float(args.weird_max_angle))
        )

        if not (near_border or small or weird):
            continue

        flags = []
        if near_border:
            flags.append("border")
        if small:
            flags.append("small")
        if weird:
            flags.append("weird")
        flags_s = ",".join(flags)

        teacher_detected = bool(sb.detected) if sb is not None else False
        teacher_conf = float(sb.conf) if sb is not None else 0.0
        teacher_iou = sb.iou if sb is not None else None
        teacher_err_mean_px = sb.err_mean_px if sb is not None else None
        teacher_err_max_px = sb.err_max_px if sb is not None else None

        teacher_match = (
            teacher_detected
            and teacher_conf >= float(args.confirm_conf)
            and (teacher_iou is not None and float(teacher_iou) >= float(args.confirm_iou))
        )
        teacher_status = "match" if teacher_match else "no_match"
        teacher_confirmed = bool(teacher_match)

        sb_status_counts[teacher_status] = sb_status_counts.get(teacher_status, 0) + 1

        cand = Candidate(
            split=split,
            filename=filename,
            coords=coords,
            w=w,
            h=h,
            border_margin_norm=float(border_margin_norm),
            border_margin_px=float(border_margin_px) if border_margin_px is not None else None,
            area_ratio=float(area_ratio),
            bbox_area_ratio=float(bbox_area),
            bbox_w_norm=float(bbox_w),
            bbox_h_norm=float(bbox_h),
            min_dim_norm=float(min_dim_norm),
            fill_ratio=float(fill_ratio),
            self_intersect_raw=bool(self_intersect_raw),
            convex=bool(convex),
            min_angle=float(min_angle),
            max_angle=float(max_angle),
            edge_ratio=float(edge_ratio),
            near_border=bool(near_border),
            small=bool(small),
            weird=bool(weird),
            flags=flags_s,
            teacher_detected=bool(teacher_detected),
            teacher_conf=float(teacher_conf),
            teacher_iou=float(teacher_iou) if teacher_iou is not None else None,
            teacher_err_mean_px=float(teacher_err_mean_px) if teacher_err_mean_px is not None else None,
            teacher_err_max_px=float(teacher_err_max_px) if teacher_err_max_px is not None else None,
            teacher_status=str(teacher_status),
            teacher_match=bool(teacher_match),
            teacher_confirmed=bool(teacher_confirmed),
        )
        candidates.append(cand)

        if near_border:
            counts["near_border"] += 1
        if small:
            counts["small"] += 1
        if weird:
            counts["weird"] += 1
        counts["any"] += 1
        if teacher_confirmed:
            counts["confirmed"] += 1

    # Write CSVs
    fields = [
        "split",
        "filename",
        "flags",
        "near_border",
        "small",
        "weird",
        "border_margin_norm",
        "border_margin_px",
        "area_ratio",
        "bbox_area_ratio",
        "bbox_w_norm",
        "bbox_h_norm",
        "min_dim_norm",
        "fill_ratio",
        "self_intersect_raw",
        "convex",
        "min_angle",
        "max_angle",
        "edge_ratio",
        "teacher_detected",
        "teacher_conf",
        "teacher_iou",
        "teacher_err_mean_px",
        "teacher_err_max_px",
        "teacher_status",
        "teacher_match",
        "teacher_confirmed",
    ]

    candidates_csv = out_dir / "candidates.csv"
    confirmed_csv = out_dir / "exclude_candidates.csv"

    with candidates_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in candidates:
            w.writerow({k: getattr(c, k) for k in fields})

    confirmed = [c for c in candidates if c.teacher_confirmed]
    with confirmed_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in confirmed:
            w.writerow({k: getattr(c, k) for k in fields})

    # Exclusion candidate lists per split (NOT applied automatically)
    excl_train = sorted({c.filename for c in confirmed if c.split == "train"})
    excl_val = sorted({c.filename for c in confirmed if c.split == "val"})
    _write_lines(out_dir / "exclude_candidates_train.txt", excl_train)
    _write_lines(out_dir / "exclude_candidates_val.txt", excl_val)
    _write_lines(out_dir / "exclude_candidates_all.txt", sorted({c.filename for c in confirmed}))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Candidates (GT-flagged): {counts['any']}")
    print(f"  near_border: {counts['near_border']}")
    print(f"  small:       {counts['small']}")
    print(f"  weird:       {counts['weird']}")
    print(f"Confirmed by teacher: conf>={args.confirm_conf} and IoU>={args.confirm_iou} -> {counts['confirmed']}")
    if sb_status_counts:
        print("Teacher status breakdown (candidates):")
        for k in sorted(sb_status_counts.keys()):
            print(f"  {k}: {sb_status_counts[k]}")
    print(f"Candidates CSV: {candidates_csv}")
    print(f"Exclude candidates CSV: {confirmed_csv}")

    if args.make_collage:
        # Load teacher coords from <split>_teacher.txt (generated once via generate_teacher_gt_splits.py)
        teacher_coords_by_split: dict[str, dict[str, np.ndarray]] = {}
        for split in splits:
            teacher_split_file = data_root / f"{split}_teacher.txt"
            if not teacher_split_file.exists():
                raise SystemExit(
                    f"Missing {teacher_split_file}. Run: python generate_teacher_gt_splits.py --data_root {data_root} --overwrite"
                )
            teacher_coords_by_split[split] = _load_teacher_split_file(teacher_split_file)

        # Sort confirmed for more variety: by flags then by border/small/area
        def sort_key(c: Candidate) -> tuple:
            # Put "weird" first, then border, then small; within, prioritize extreme values
            prio = 0
            if c.weird:
                prio = 0
            elif c.near_border:
                prio = 1
            else:
                prio = 2
            return (prio, c.border_margin_norm, c.area_ratio)

        confirmed_sorted = sorted(confirmed, key=sort_key)
        out_paths = _make_collages(
            candidates=confirmed_sorted,
            data_root=data_root,
            out_dir=out_dir,
            tiles_per_page=int(args.tiles_per_page),
            cols=int(args.cols),
            tile_size=int(args.tile_size),
            collage_format=str(args.collage_format),
            jpeg_quality=int(args.jpeg_quality),
            teacher_coords_by_split=teacher_coords_by_split,
        )
        if out_paths:
            print(f"Collages: {out_dir / 'collages'} ({len(out_paths)} pages)")

    print("\nDone.")


if __name__ == "__main__":
    main()
