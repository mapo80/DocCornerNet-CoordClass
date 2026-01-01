#!/usr/bin/env python3
"""
Build a final training dataset via 4-way automatic "review":
  - GT (labels/*.txt)
  - External teacher model (<split>_<teacher_suffix>.txt)
  - Winner model (train_winner.txt / val_winner.txt)
  - DocAligner (train_docaligner.txt / val_docaligner.txt)

Outputs:
  - <data_root>/train_final.txt
  - <data_root>/val_final.txt

Optionally updates label files in-place (with backup):
  - <data_root>/labels/<stem>.txt
  - backups stored in <data_root>/labels_backup_consensus4_<timestamp>/

Hard constraints:
  - Keep negatives in the final split files.
  - Never drop images whose stem starts with "MIDV" or "smartdoc" (case-insensitive).
  - Ensure final train+val size >= --min_total (default 60000). If dropping would violate,
    drops are automatically disabled (kept as-is).

Heuristics (default thresholds are conservative):
  - If GT vs teacher IoU >= --teacher_win_iou, pick teacher ("teacher wins at very high IoU").
  - If a non-GT cluster (>=2 among Teacher/Winner/DocAligner) agrees at --agree_iou and GT
    disagrees (IoU <= --gt_disagree_iou), replace GT with the preferred source from that cluster.
  - If no agreement and strong multi-way disagreement, optionally drop the sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class Pred:
    detected: int
    conf: float
    coords: Optional[np.ndarray]  # [8] normalized, TL/TR/BR/BL-ish (may be reordered later)


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


def _load_pred_file(path: Path) -> dict[str, Pred]:
    """
    Load a <split>_*.txt file with 11 tokens:
      filename detected conf x0 y0 x1 y1 x2 y2 x3 y3
    """
    out: dict[str, Pred] = {}
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
                conf = float(parts[2])
                coords = np.array([float(x) for x in parts[3:11]], dtype=np.float32)
            except Exception:
                continue
            if coords.shape != (8,) or not np.all(np.isfinite(coords)):
                continue
            if detected != 1:
                out[name] = Pred(detected=0, conf=conf, coords=None)
                continue
            if np.any(coords < 0.0):
                out[name] = Pred(detected=0, conf=conf, coords=None)
                continue
            out[name] = Pred(detected=1, conf=conf, coords=np.clip(coords, 0.0, 1.0))
    return out


def _load_yolo_polygon(label_path: Path) -> Optional[np.ndarray]:
    """
    YOLO polygon format: class x0 y0 x1 y1 x2 y2 x3 y3
    Returns coords [8] or None if invalid.
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
    if coords.shape != (8,) or not np.all(np.isfinite(coords)):
        return None
    coords = np.clip(coords, 0.0, 1.0)
    return coords


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


def _order_points_ccw(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(angles)]


def _poly_from_quad(coords8: np.ndarray):
    from shapely.geometry import Polygon
    from shapely.validation import make_valid

    pts = coords8.reshape(4, 2)
    pts = _order_points_ccw(pts)
    try:
        poly = Polygon([(float(x), float(y)) for x, y in pts])
    except Exception:
        return None

    if poly.is_empty:
        return None

    if not poly.is_valid:
        try:
            poly2 = make_valid(poly)
        except Exception:
            return None
        if poly2.is_empty:
            return None
        if poly2.geom_type == "Polygon":
            poly = poly2
        elif poly2.geom_type == "MultiPolygon":
            poly = max(poly2.geoms, key=lambda p: p.area)
        elif poly2.geom_type == "GeometryCollection":
            polys = [g for g in poly2.geoms if g.geom_type == "Polygon"]
            poly = max(polys, key=lambda p: p.area) if polys else None
            if poly is None:
                return None
        else:
            return None

    if poly.area <= 0.0:
        return None
    return poly


def _iou(poly_a, poly_b) -> float:
    inter = float(poly_a.intersection(poly_b).area)
    union = float(poly_a.union(poly_b).area)
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _is_protected(filename: str) -> bool:
    stem = Path(filename).stem
    if stem.startswith("MIDV"):
        return True
    if stem.lower().startswith("smartdoc"):
        return True
    return False


def _best_component(sources: list[str], ious: dict[tuple[str, str], float], thr: float) -> list[str]:
    # Build adjacency
    adj: dict[str, set[str]] = {s: set() for s in sources}
    for i, a in enumerate(sources):
        for b in sources[i + 1 :]:
            v = float(ious.get((a, b), ious.get((b, a), 0.0)))
            if v >= float(thr):
                adj[a].add(b)
                adj[b].add(a)

    # Connected components
    seen: set[str] = set()
    comps: list[list[str]] = []
    for s in sources:
        if s in seen:
            continue
        stack = [s]
        comp: list[str] = []
        seen.add(s)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(comp)

    def comp_score(comp: list[str]) -> tuple[int, float]:
        # size first, then mean IoU of internal edges
        if len(comp) <= 1:
            return (len(comp), 0.0)
        vals: list[float] = []
        for i, a in enumerate(comp):
            for b in comp[i + 1 :]:
                vals.append(float(ious.get((a, b), ious.get((b, a), 0.0))))
        return (len(comp), float(np.mean(vals) if vals else 0.0))

    comps.sort(key=lambda c: comp_score(c), reverse=True)
    return comps[0] if comps else []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build final train/val splits and (optionally) fix labels via 4-way consensus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new",
        help="Dataset root (contains images/, images-negative/, labels/, train.txt, val.txt).",
    )
    p.add_argument("--splits", type=str, default="train,val", help="Comma-separated split basenames")
    p.add_argument("--teacher_suffix", type=str, default="teacher", help="Suffix for <split>_<suffix>.txt")
    p.add_argument("--winner_suffix", type=str, default="winner", help="Suffix for <split>_<suffix>.txt")
    p.add_argument("--docaligner_suffix", type=str, default="docaligner", help="Suffix for <split>_<suffix>.txt")
    p.add_argument("--out_train", type=str, default="train_final.txt", help="Output train split filename")
    p.add_argument("--out_val", type=str, default="val_final.txt", help="Output val split filename")
    p.add_argument("--report_csv", type=str, default="consensus4_decisions.csv", help="Per-image decisions CSV")
    p.add_argument("--summary_json", type=str, default="consensus4_summary.json", help="Summary JSON")
    p.add_argument("--backup_dir", type=str, default="", help="Backup dir for modified labels (default: auto)")
    p.add_argument("--dry_run", action="store_true", help="Do not modify labels, only write outputs/reports")
    p.add_argument(
        "--full_report",
        action="store_true",
        help="Compute and write all pairwise IoUs in the CSV (slower).",
    )

    # Thresholds / rules
    p.add_argument(
        "--teacher_win_iou",
        type=float,
        default=0.995,
        help="If IoU(GT,teacher) >= this -> choose teacher (very high IoU preference)",
    )
    p.add_argument("--agree_iou", type=float, default=0.95, help="IoU threshold for non-GT agreement cluster")
    p.add_argument("--gt_disagree_iou", type=float, default=0.85, help="If IoU(GT,chosen) <= this -> replace GT")
    p.add_argument("--drop_if_max_iou_below", type=float, default=0.50, help="Drop if max pairwise IoU < this")
    p.add_argument("--min_total", type=int, default=60000, help="Minimum total (train+val) kept entries incl negatives")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]

    labels_dir = data_root / "labels"
    if not labels_dir.exists():
        raise SystemExit(f"Missing labels dir: {labels_dir}")

    out_paths: dict[str, Path] = {}
    for s in splits:
        if s == "train":
            out_paths[s] = data_root / str(args.out_train)
        elif s == "val":
            out_paths[s] = data_root / str(args.out_val)
        else:
            out_paths[s] = data_root / f"{s}_final.txt"

    backup_dir = Path(args.backup_dir) if args.backup_dir else data_root / f"labels_backup_consensus4_{_timestamp()}"
    if not args.dry_run:
        backup_dir.mkdir(parents=True, exist_ok=True)

    report_path = data_root / str(args.report_csv)
    summary_path = data_root / str(args.summary_json)

    print("\n" + "=" * 80)
    print("Consensus4 dataset builder")
    print("=" * 80)
    print(f"Dataset: {data_root}")
    print(f"Splits: {splits}")
    print(f"Outputs: {', '.join(str(p.name) for p in out_paths.values())}")
    print(f"dry_run={bool(args.dry_run)}  backup_dir={backup_dir if not args.dry_run else '(dry-run)'}")
    print(
        f"teacher_win_iou={args.teacher_win_iou} agree_iou={args.agree_iou} "
        f"gt_disagree_iou={args.gt_disagree_iou} drop_if_max_iou_below={args.drop_if_max_iou_below}"
    )

    # Pre-load all pred maps (per split) once
    pred_maps: dict[tuple[str, str], dict[str, Pred]] = {}
    for split in splits:
        for kind, suffix in [
            ("teacher", args.teacher_suffix),
            ("winner", args.winner_suffix),
            ("docaligner", args.docaligner_suffix),
        ]:
            p = data_root / f"{split}_{suffix}.txt"
            if not p.exists():
                raise SystemExit(f"Missing {kind} file: {p}")
            pred_maps[(split, kind)] = _load_pred_file(p)

    fields = [
        "split",
        "filename",
        "protected",
        "action",
        "chosen_source",
        "teacher_conf",
        "winner_score",
        "docaligner_conf",
        "iou_gt_teacher",
        "iou_gt_winner",
        "iou_gt_docaligner",
        "iou_teacher_winner",
        "iou_teacher_docaligner",
        "iou_winner_docaligner",
        "max_pairwise_iou",
        "reason",
    ]

    counts = {
        "total": 0,
        "positives": 0,
        "negatives": 0,
        "kept": 0,
        "dropped": 0,
        "changed_labels": 0,
        "kept_protected": 0,
        "dropped_attempted_protected": 0,
    }

    # First pass: decide + (optionally) modify labels, write split outputs and report.
    kept_names_by_split: dict[str, list[str]] = {s: [] for s in splits}
    dropped_names: list[str] = []
    changed_labels: list[str] = []

    with report_path.open("w", newline="") as rf:
        writer = csv.DictWriter(rf, fieldnames=fields)
        writer.writeheader()

        for split in splits:
            split_file = data_root / f"{split}.txt"
            if not split_file.exists():
                raise SystemExit(f"Missing split file: {split_file}")

            names = _read_split_lines(split_file)
            for filename in tqdm(names, desc=f"Consensus ({split})"):
                counts["total"] += 1

                is_neg = filename.startswith("negative_")
                protected = _is_protected(filename)

                if is_neg:
                    counts["negatives"] += 1
                    counts["kept"] += 1
                    kept_names_by_split[split].append(filename)
                    writer.writerow(
                        {
                            "split": split,
                            "filename": filename,
                            "protected": int(protected),
                            "action": "keep_negative",
                            "chosen_source": "none",
                            "teacher_conf": "",
                            "winner_score": "",
                            "docaligner_conf": "",
                            "iou_gt_teacher": "",
                            "iou_gt_winner": "",
                            "iou_gt_docaligner": "",
                            "iou_teacher_winner": "",
                            "iou_teacher_docaligner": "",
                            "iou_winner_docaligner": "",
                            "max_pairwise_iou": "",
                            "reason": "negative",
                        }
                    )
                    continue

                counts["positives"] += 1

                gt_path = labels_dir / (Path(filename).stem + ".txt")
                gt_coords = _load_yolo_polygon(gt_path)
                if gt_coords is None:
                    # should not happen in this dataset; keep as negative fallback
                    counts["kept"] += 1
                    kept_names_by_split[split].append(filename)
                    writer.writerow(
                        {
                            "split": split,
                            "filename": filename,
                            "protected": int(protected),
                            "action": "keep_missing_gt",
                            "chosen_source": "gt_missing",
                            "teacher_conf": "",
                            "winner_score": "",
                            "docaligner_conf": "",
                            "iou_gt_teacher": "",
                            "iou_gt_winner": "",
                            "iou_gt_docaligner": "",
                            "iou_teacher_winner": "",
                            "iou_teacher_docaligner": "",
                            "iou_winner_docaligner": "",
                            "max_pairwise_iou": "",
                            "reason": "missing_gt_label",
                        }
                    )
                    continue

                # Build candidate polys
                source_coords: dict[str, np.ndarray] = {"gt": gt_coords}
                source_conf: dict[str, float] = {"gt": 1.0}

                teacher = pred_maps[(split, "teacher")].get(filename)
                wn = pred_maps[(split, "winner")].get(filename)
                da = pred_maps[(split, "docaligner")].get(filename)

                if teacher is not None and teacher.detected == 1 and teacher.coords is not None:
                    source_coords["teacher"] = teacher.coords
                    source_conf["teacher"] = float(teacher.conf)
                if wn is not None and wn.detected == 1 and wn.coords is not None:
                    source_coords["winner"] = wn.coords
                    source_conf["winner"] = float(wn.conf)
                if da is not None and da.detected == 1 and da.coords is not None:
                    source_coords["docaligner"] = da.coords
                    source_conf["docaligner"] = float(da.conf)

                polys: dict[str, object] = {}
                ious: dict[tuple[str, str], float] = {}
                max_iou = 0.0

                def poly_for(k: str):
                    if k in polys:
                        return polys[k]
                    c = source_coords.get(k)
                    if c is None:
                        return None
                    poly = _poly_from_quad(c)
                    if poly is None:
                        source_coords.pop(k, None)
                        source_conf.pop(k, None)
                        return None
                    polys[k] = poly
                    return poly

                def iou_for(a: str, b: str) -> Optional[float]:
                    nonlocal max_iou
                    if (a, b) in ious:
                        return ious[(a, b)]
                    if (b, a) in ious:
                        return ious[(b, a)]
                    pa = poly_for(a)
                    pb = poly_for(b)
                    if pa is None or pb is None:
                        return None
                    v = _iou(pa, pb)
                    ious[(a, b)] = v
                    if v > max_iou:
                        max_iou = v
                    return v

                def get_iou(a: str, b: str) -> str:
                    if args.full_report:
                        v = iou_for(a, b)
                        if v is None:
                            return ""
                        return f"{float(v):.6f}"
                    # Fast path: only report IoUs that were already computed for decisions.
                    if (a, b) in ious:
                        return f"{float(ious[(a, b)]):.6f}"
                    if (b, a) in ious:
                        return f"{float(ious[(b, a)]):.6f}"
                    return ""

                chosen_source = "gt"
                action = "keep_gt"
                reason = "default_keep_gt"

                iou_gt_teacher = iou_for("gt", "teacher")
                if iou_gt_teacher is not None and iou_gt_teacher >= float(args.teacher_win_iou):
                    chosen_source = "teacher"
                    action = "replace_teacher_high_iou"
                    reason = f"iou_gt_teacher>={args.teacher_win_iou}"
                else:
                    # Non-GT agreement cluster
                    non_gt = [s for s in ("teacher", "winner", "docaligner") if source_coords.get(s) is not None]
                    # Ensure polys/ious exist for non-GT pairwise comparisons
                    for s in list(non_gt):
                        if poly_for(s) is None:
                            non_gt.remove(s)
                    for i, a in enumerate(non_gt):
                        for b in non_gt[i + 1 :]:
                            _ = iou_for(a, b)

                    comp = _best_component(non_gt, ious, float(args.agree_iou)) if non_gt else []
                    if len(comp) >= 2:
                        if "teacher" in comp:
                            cand = "teacher"
                        elif "winner" in comp:
                            cand = "winner"
                        else:
                            cand = "docaligner"

                        iou_gt_cand = iou_for("gt", cand)
                        if iou_gt_cand is not None and iou_gt_cand <= float(args.gt_disagree_iou):
                            chosen_source = cand
                            action = f"replace_{cand}_cluster"
                            reason = f"cluster_{args.agree_iou}_gt_iou<={args.gt_disagree_iou}"
                        else:
                            chosen_source = "gt"
                            action = "keep_gt_cluster_not_far"
                            reason = f"cluster_{args.agree_iou}_gt_iou>{args.gt_disagree_iou}"
                    else:
                        # Drop only if strong disagreement and enough sources
                        # Materialize remaining polys + compute max IoU
                        for s in list(source_coords.keys()):
                            _ = poly_for(s)
                        keys = list(polys.keys())
                        for i, a in enumerate(keys):
                            for b in keys[i + 1 :]:
                                _ = iou_for(a, b)

                        if len(polys) >= 3 and max_iou < float(args.drop_if_max_iou_below):
                            if protected:
                                chosen_source = "gt"
                                action = "keep_gt_protected"
                                reason = "protected_no_drop"
                                counts["dropped_attempted_protected"] += 1
                            else:
                                chosen_source = "none"
                                action = "drop"
                                reason = f"max_pairwise_iou<{args.drop_if_max_iou_below}"

                # Apply action
                if action == "drop":
                    counts["dropped"] += 1
                    dropped_names.append(filename)
                else:
                    counts["kept"] += 1
                    kept_names_by_split[split].append(filename)
                    if protected:
                        counts["kept_protected"] += 1

                    if chosen_source != "gt" and chosen_source in source_coords:
                        new_coords = _order_points_tl_tr_br_bl(np.asarray(source_coords[chosen_source], dtype=np.float32))
                        new_coords = np.clip(new_coords, 0.0, 1.0)

                        if not args.dry_run:
                            # backup original label once
                            dst = backup_dir / gt_path.name
                            if not dst.exists():
                                shutil.copy2(gt_path, dst)
                            # write new label
                            line = "0 " + " ".join(f"{float(v):.6f}" for v in new_coords.tolist()) + "\n"
                            gt_path.write_text(line)

                        counts["changed_labels"] += 1
                        changed_labels.append(filename)

                writer.writerow(
                    {
                        "split": split,
                        "filename": filename,
                        "protected": int(protected),
                        "action": action,
                        "chosen_source": chosen_source,
                        "teacher_conf": f"{float(source_conf.get('teacher', 0.0)):.6f}" if "teacher" in source_conf else "",
                        "winner_score": f"{float(source_conf.get('winner', 0.0)):.6f}" if "winner" in source_conf else "",
                        "docaligner_conf": f"{float(source_conf.get('docaligner', 0.0)):.6f}"
                        if "docaligner" in source_conf
                        else "",
                        "iou_gt_teacher": get_iou("gt", "teacher"),
                        "iou_gt_winner": get_iou("gt", "winner"),
                        "iou_gt_docaligner": get_iou("gt", "docaligner"),
                        "iou_teacher_winner": get_iou("teacher", "winner"),
                        "iou_teacher_docaligner": get_iou("teacher", "docaligner"),
                        "iou_winner_docaligner": get_iou("winner", "docaligner"),
                        "max_pairwise_iou": f"{float(max_iou):.6f}" if len(polys) >= 2 else "",
                        "reason": reason,
                    }
                )

    # Enforce min_total: if too many drops, disable drops by keeping them (no label changes).
    total_kept = sum(len(v) for v in kept_names_by_split.values())
    if int(total_kept) < int(args.min_total):
        print(
            f"\nWARNING: kept={total_kept} < min_total={args.min_total}. "
            "Disabling drops (keeping dropped samples as-is)."
        )
        for split in splits:
            split_file = data_root / f"{split}.txt"
            names = _read_split_lines(split_file)
            kept_names_by_split[split] = names[:]  # keep everything
        counts["kept"] = sum(len(v) for v in kept_names_by_split.values())
        counts["dropped"] = 0
        dropped_names = []

    # Write final split files
    for split, out_path in out_paths.items():
        out_path.write_text("\n".join(kept_names_by_split[split]) + "\n")

    # Write drop list / changed list for convenience
    (data_root / "consensus4_dropped.txt").write_text("\n".join(dropped_names) + ("\n" if dropped_names else ""))
    (data_root / "consensus4_labels_changed.txt").write_text(
        "\n".join(changed_labels) + ("\n" if changed_labels else "")
    )

    # Summary
    summary = {
        "data_root": str(data_root),
        "splits": splits,
        "outputs": {k: str(v) for k, v in out_paths.items()},
        "dry_run": bool(args.dry_run),
        "backup_dir": "" if args.dry_run else str(backup_dir),
        "thresholds": {
            "teacher_win_iou": float(args.teacher_win_iou),
            "agree_iou": float(args.agree_iou),
            "gt_disagree_iou": float(args.gt_disagree_iou),
            "drop_if_max_iou_below": float(args.drop_if_max_iou_below),
            "min_total": int(args.min_total),
        },
        "counts": counts,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\nDone.")
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")
    for split, out_path in out_paths.items():
        print(f"{split}: {len(kept_names_by_split[split])} -> {out_path.name}")
    if not args.dry_run:
        print(f"Labels changed: {len(changed_labels)} (backup: {backup_dir})")


if __name__ == "__main__":
    main()
