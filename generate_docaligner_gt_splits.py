#!/usr/bin/env python3
"""
Generate DocAligner pseudo-labels (corners) for dataset splits to support multi-model consensus.

Writes:
  - <data_root>/train_docaligner.txt
  - <data_root>/val_docaligner.txt

Each line has a fixed format (11 tokens):
  filename detected confidence x0 y0 x1 y1 x2 y2 x3 y3

Notes:
  - It preserves split file order and includes negatives as detected=0 with coords=-1.
  - It does NOT modify labels/ nor train.txt/val.txt.
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

_G_ALIGNER = None


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


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


def _init_worker(docaligner_repo: str, model_type: str, model_cfg: str, backend: str, gpu_id: int) -> None:
    global _G_ALIGNER

    repo_path = Path(docaligner_repo)
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    from docaligner import DocAligner, ModelType  # type: ignore

    # backend is capybara.Backend enum; accept 'cpu'/'gpu' strings
    import capybara as cb  # type: ignore

    if str(backend).lower() == "gpu":
        cb_backend = cb.Backend.gpu
    else:
        cb_backend = cb.Backend.cpu

    if str(model_type).lower() in {"point", "pt"}:
        mt = ModelType.point
    else:
        mt = ModelType.heatmap

    _G_ALIGNER = DocAligner(model_type=mt, model_cfg=model_cfg, backend=cb_backend, gpu_id=int(gpu_id))


@dataclass(frozen=True)
class Task:
    filename: str
    is_negative: bool
    data_root: str


def _process_one(task: Task) -> tuple[str, int, float, np.ndarray]:
    if _G_ALIGNER is None:
        raise RuntimeError("Worker not initialized")

    if task.is_negative:
        return task.filename, 0, 0.0, np.full((8,), -1.0, dtype=np.float32)

    data_root = Path(task.data_root)
    img_path = data_root / "images" / task.filename
    if not img_path.exists():
        return task.filename, 0, 0.0, np.full((8,), -1.0, dtype=np.float32)

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            arr = np.asarray(img, dtype=np.uint8)

        poly = _G_ALIGNER(arr, do_center_crop=False)
        poly = np.asarray(poly)
        if poly.ndim != 2 or poly.shape != (4, 2):
            return task.filename, 0, 0.0, np.full((8,), -1.0, dtype=np.float32)

        # Normalize to [0..1]
        pts = poly.astype(np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0.0, float(w - 1))
        pts[:, 1] = np.clip(pts[:, 1], 0.0, float(h - 1))
        pts[:, 0] = pts[:, 0] / float(w)
        pts[:, 1] = pts[:, 1] / float(h)
        coords = pts.reshape(-1)
        coords = np.clip(coords, 0.0, 1.0)
        coords = _order_points_tl_tr_br_bl(coords)

        # DocAligner doesn't expose a confidence here; keep 1.0 for detected.
        return task.filename, 1, 1.0, coords.astype(np.float32)
    except Exception:
        return task.filename, 0, 0.0, np.full((8,), -1.0, dtype=np.float32)


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate DocAligner pseudo-label split files (train_docaligner.txt/val_docaligner.txt).",
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
        "--docaligner_repo",
        type=str,
        default="/Volumes/ZX20/ML-Models/DocScannerDetection/third-party/DocAligner",
        help="Path to DocAligner repo (for docaligner import).",
    )
    p.add_argument("--model_type", type=str, default="point", choices=["point", "heatmap"], help="DocAligner model type")
    p.add_argument(
        "--model_cfg",
        type=str,
        default="lcnet050",
        help="DocAligner model config (point: lcnet050, heatmap: fastvit_sa24/fastvit_t8/lcnet100).",
    )
    p.add_argument("--backend", type=str, default="cpu", choices=["cpu", "gpu"], help="DocAligner backend")
    p.add_argument("--gpu_id", type=int, default=0, help="GPU id (if backend=gpu)")
    p.add_argument("--workers", type=int, default=min(4, (os.cpu_count() or 4)), help="Process workers")
    p.add_argument("--chunksize", type=int, default=16, help="ProcessPool chunksize")
    p.add_argument("--max_images", type=int, default=0, help="If >0, limit per split (debug)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing <split>_docaligner.txt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]

    # Pre-load once in main process to ensure model weights are present (avoids concurrent downloads).
    repo_path = Path(args.docaligner_repo)
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    try:
        from docaligner import DocAligner, ModelType  # type: ignore
        import capybara as cb  # type: ignore

        mt = ModelType.point if str(args.model_type).lower() == "point" else ModelType.heatmap
        cb_backend = cb.Backend.cpu if str(args.backend).lower() == "cpu" else cb.Backend.gpu
        _ = DocAligner(model_type=mt, model_cfg=str(args.model_cfg), backend=cb_backend, gpu_id=int(args.gpu_id))
    except Exception as e:
        raise SystemExit(f"Failed to initialize DocAligner from {repo_path}: {e}")

    print("\n" + "=" * 80)
    print("Generating DocAligner GT split files")
    print("=" * 80)
    print(f"Dataset: {data_root}")
    print(f"Splits: {splits}")
    print(f"Repo: {repo_path}")
    print(f"Model: {args.model_type}:{args.model_cfg} backend={args.backend}")
    print(f"Workers: {args.workers}")

    with ProcessPoolExecutor(
        max_workers=int(args.workers),
        initializer=_init_worker,
        initargs=(str(repo_path), str(args.model_type), str(args.model_cfg), str(args.backend), int(args.gpu_id)),
    ) as ex:
        for split in splits:
            split_file = data_root / f"{split}.txt"
            if not split_file.exists():
                raise SystemExit(f"Missing split file: {split_file}")

            out_path = data_root / f"{split}_docaligner.txt"
            if out_path.exists() and not bool(args.overwrite):
                raise SystemExit(f"Refusing to overwrite existing {out_path}. Use --overwrite.")

            names = _read_split_lines(split_file)
            if args.max_images and args.max_images > 0:
                names = names[: int(args.max_images)]

            tasks = [Task(filename=n, is_negative=n.startswith("negative_"), data_root=str(data_root)) for n in names]
            print(f"\n{split}: {len(tasks)} entries -> {out_path.name}")

            with out_path.open("w") as f:
                for filename, detected, conf, coords in tqdm(
                    ex.map(_process_one, tasks, chunksize=int(args.chunksize)),
                    total=len(tasks),
                    desc=f"DocAligner ({split})",
                ):
                    parts = [filename, str(int(detected)), f"{float(conf):.6f}"] + [
                        f"{float(x):.6f}" for x in coords.tolist()
                    ]
                    f.write(" ".join(parts) + "\n")

    print("\nDone.")


if __name__ == "__main__":
    main()
