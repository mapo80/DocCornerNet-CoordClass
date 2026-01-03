#!/usr/bin/env python3
"""
Generate teacher "GT" (pseudo-labels) for dataset splits to avoid re-running inference.

This writes:
  - <data_root>/train_teacher.txt
  - <data_root>/val_teacher.txt

Each line corresponds to ONE image listed in the split file and has a fixed format:

  filename detected confidence x0 y0 x1 y1 x2 y2 x3 y3

Where:
  - detected: 1 if the teacher returned corners, else 0
  - confidence: teacher confidence (float)
  - xi, yi: normalized coords in [0..1] (if detected), otherwise -1

Notes:
  - It processes BOTH positives (images/) and negatives (images-negative/) if present in split file.
  - It does NOT modify train.txt / val.txt.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True

_G_DETECTOR = None
_G_DATA_ROOT: Optional[Path] = None


def _init_worker(teacher_repo: str, backbone_path: str, head_path: str, data_root: str) -> None:
    global _G_DETECTOR, _G_DATA_ROOT
    _G_DATA_ROOT = Path(data_root)

    teacher_repo_path = Path(teacher_repo)
    if str(teacher_repo_path) not in sys.path:
        sys.path.insert(0, str(teacher_repo_path))

    from document_detector import DocumentDetector  # type: ignore

    _G_DETECTOR = DocumentDetector(
        backbone_path=backbone_path,
        head_path=head_path,
        verbose=False,
    )


@dataclass(frozen=True)
class Task:
    filename: str
    is_negative: bool


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


def _process_one(task: Task) -> tuple[str, int, float, np.ndarray]:
    if _G_DETECTOR is None or _G_DATA_ROOT is None:
        raise RuntimeError("Worker not initialized")

    img_dir = _G_DATA_ROOT / ("images-negative" if task.is_negative else "images")
    img_path = img_dir / task.filename

    if not img_path.exists():
        return task.filename, 0, 0.0, np.full((8,), -1.0, dtype=np.float32)

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            w, h = img.size

            res = _G_DETECTOR.detect(img)
            conf = float(res.get("confidence", 0.0) or 0.0)
            corners = res.get("corners", None)

            if corners is None:
                return task.filename, 0, conf, np.full((8,), -1.0, dtype=np.float32)

            flat: list[float] = []
            for x, y in corners:
                flat.append(float(x) / float(w))
                flat.append(float(y) / float(h))
            if len(flat) != 8:
                return task.filename, 0, conf, np.full((8,), -1.0, dtype=np.float32)

            coords = np.array(flat, dtype=np.float32)
            coords = np.clip(coords, 0.0, 1.0)
            return task.filename, 1, conf, coords
    except Exception:
        return task.filename, 0, 0.0, np.full((8,), -1.0, dtype=np.float32)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate teacher pseudo-label split files (train_teacher.txt/val_teacher.txt).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new",
        help="Dataset root (contains images/, images-negative/, train.txt, val.txt).",
    )
    p.add_argument("--splits", type=str, default="train,val", help="Comma-separated split basenames")
    p.add_argument(
        "--teacher_repo",
        type=str,
        default="",
        help="Path to a teacher detector repo (must provide document_detector.DocumentDetector).",
    )
    p.add_argument(
        "--teacher_backbone",
        type=str,
        default="",
        help="Teacher backbone model path (passed to DocumentDetector).",
    )
    p.add_argument(
        "--teacher_head",
        type=str,
        default="",
        help="Teacher head model path (passed to DocumentDetector).",
    )
    p.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 4)), help="Process workers")
    p.add_argument("--chunksize", type=int, default=32, help="ProcessPool chunksize")
    p.add_argument("--max_images", type=int, default=0, help="If >0, limit per split (debug)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing <split>_teacher.txt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]

    print("\n" + "=" * 80)
    print("Generating teacher GT split files")
    print("=" * 80)
    print(f"Dataset: {data_root}")
    print(f"Splits: {splits}")
    print(f"Workers: {args.workers}")

    if not str(args.teacher_repo).strip():
        raise SystemExit("Missing --teacher_repo (path to a repo that provides document_detector.DocumentDetector).")
    if not str(args.teacher_backbone).strip():
        raise SystemExit("Missing --teacher_backbone")
    if not str(args.teacher_head).strip():
        raise SystemExit("Missing --teacher_head")

    with ProcessPoolExecutor(
        max_workers=int(args.workers),
        initializer=_init_worker,
        initargs=(args.teacher_repo, args.teacher_backbone, args.teacher_head, str(data_root)),
    ) as ex:
        for split in splits:
            split_file = data_root / f"{split}.txt"
            if not split_file.exists():
                raise SystemExit(f"Missing split file: {split_file}")

            out_path = data_root / f"{split}_teacher.txt"
            if out_path.exists() and not bool(args.overwrite):
                raise SystemExit(f"Refusing to overwrite existing {out_path}. Use --overwrite.")

            names = _read_split_lines(split_file)
            if args.max_images and args.max_images > 0:
                names = names[: int(args.max_images)]

            tasks = [Task(filename=n, is_negative=n.startswith("negative_")) for n in names]
            print(f"\n{split}: {len(tasks)} images -> {out_path.name}")

            with out_path.open("w") as f:
                for filename, detected, conf, coords in tqdm(
                    ex.map(_process_one, tasks, chunksize=int(args.chunksize)),
                    total=len(tasks),
                    desc=f"Teacher ({split})",
                ):
                    # fixed 11 tokens per line
                    parts = [filename, str(int(detected)), f"{float(conf):.6f}"] + [
                        f"{float(x):.6f}" for x in coords.tolist()
                    ]
                    f.write(" ".join(parts) + "\n")

    print("\nDone.")


if __name__ == "__main__":
    main()
