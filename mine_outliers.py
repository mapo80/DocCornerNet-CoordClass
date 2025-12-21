"""
Mine hard examples (outliers) from a dataset split using a trained DocCornerNet model.

This produces a text file (one image name per line) compatible with --outlier_list
in train.py / train_student.py.
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dataset import create_dataset, load_split_file, preload_images_to_cache
from evaluate import load_model
from metrics import compute_corner_error, compute_polygon_iou


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mine hard examples / outliers from a split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model_path", type=str, required=True, help="Model directory/.keras or .weights.h5")
    parser.add_argument(
        "--data_root",
        type=str,
        default="../../datasets/official/doc-scanner-dataset-labeled",
        help="Path to dataset root",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--input_norm",
        type=str,
        default="imagenet",
        choices=["imagenet", "zero_one", "raw255"],
        help="Must match how the checkpoint was trained.",
    )
    parser.add_argument("--negative_dir", type=str, default="images-negative")

    # Optional caching / fast mode (same pattern as train.py)
    parser.add_argument("--cache_images", action="store_true", help="Pre-load images into RAM")
    parser.add_argument("--cache_dir", type=str, default=None, help="Persistent disk cache dir (optional)")
    parser.add_argument("--force_cache", action="store_true", help="Force cache regeneration")
    parser.add_argument("--fast_mode", action="store_true", help="Use tensor cache + GPU augmentations (requires cache)")

    # Outlier criteria
    parser.add_argument("--iou_threshold", type=float, default=0.95, help="Select positives with IoU < threshold")
    parser.add_argument(
        "--corner_err_threshold_px",
        type=float,
        default=None,
        help="Also select positives with corner error > threshold (in pixels).",
    )
    parser.add_argument("--score_threshold", type=float, default=0.5, help="Score threshold for FP/FN mining")
    parser.add_argument("--include_fp", action="store_true", help="Include negatives predicted as positives (FP)")
    parser.add_argument("--include_fn", action="store_true", help="Include positives predicted as negatives (FN)")
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="If >0, ignore iou_threshold and output the K lowest-IoU positives (plus optional FP/FN).",
    )

    parser.add_argument("--max_samples", type=int, default=0, help="If >0, stop after N samples (debug)")
    parser.add_argument("--output", type=str, required=True, help="Output .txt path (one image name per line)")

    # Model config (used only when loading weights without config.json)
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenetv3_small",
        choices=["mobilenetv2", "mobilenetv3_small", "mobilenetv3_large"],
    )
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--backbone_minimalistic", action="store_true")
    parser.add_argument("--backbone_include_preprocessing", action="store_true")
    parser.add_argument("--backbone_weights", type=str, default=None)
    parser.add_argument("--fpn_ch", type=int, default=48)
    parser.add_argument("--simcc_ch", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_bins", type=int, default=224)
    parser.add_argument("--tau", type=float, default=1.0)

    return parser.parse_args()


def _find_split_file(data_root: Path, split: str) -> Path:
    for prefix in (f"{split}_with_negative_v2", f"{split}_with_negative", split):
        candidate = data_root / f"{prefix}.txt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No split file found for split='{split}' in {data_root}")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main():
    args = parse_args()

    print("=" * 80)
    print("DocCornerNet Outlier Mining")
    print("=" * 80)

    # Load model (reuse evaluate.py loader so config.json is honored)
    model, img_size, _ = load_model(args)
    print(f"Model parameters: {model.count_params():,}")
    print(f"Using img_size: {img_size}")
    print(f"Input normalization: {args.input_norm}")

    data_root = Path(args.data_root)
    split_file = _find_split_file(data_root, args.split)
    image_list = load_split_file(str(split_file))
    print(f"Split file: {split_file} ({len(image_list)} samples)")

    # Optional caching
    shared_cache = None
    if args.cache_images:
        image_dir = data_root / "images"
        negative_dir = data_root / args.negative_dir
        shared_cache = preload_images_to_cache(
            image_list=image_list,
            image_dir=image_dir,
            negative_dir=negative_dir,
            img_size=img_size,
            cache_dir=args.cache_dir,
            force_cache=args.force_cache,
        )

    use_fast_mode = args.fast_mode and shared_cache is not None
    if args.fast_mode and shared_cache is None:
        print("Warning: --fast_mode requires --cache_images, falling back to standard mode")
    if use_fast_mode:
        print("⚡ Using FAST MODE (cached tensors)")

    ds = create_dataset(
        data_root=args.data_root,
        split=args.split,
        img_size=img_size,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
        negative_dir=args.negative_dir,
        shared_cache=shared_cache,
        fast_mode=use_fast_mode,
        drop_remainder=False,
        image_norm=args.input_norm,
    )

    hard_pos = []  # (iou, name, err_px, score)
    hard_fp = []   # (name, score)
    hard_fn = []   # (name, score)

    n_seen = 0
    n_pos = 0
    n_neg = 0

    for images, targets in tqdm(ds, desc="Mining", unit="batch"):
        outputs = model(images, training=False)

        # Support training model (dict) and inference model (tuple/list)
        if isinstance(outputs, dict):
            coords_pred = outputs["coords"].numpy()
            score_logit = outputs["score_logit"].numpy()
        else:
            coords_pred = outputs[0].numpy()
            score_logit = outputs[1].numpy()

        score = _sigmoid(score_logit.squeeze(-1))
        coords_gt = targets["coords"].numpy()
        has_doc = targets["has_doc"].numpy().astype(np.float32)
        if has_doc.ndim == 2:
            has_doc = has_doc.squeeze(-1)

        batch_size = int(coords_gt.shape[0])
        for i in range(batch_size):
            if args.max_samples and n_seen >= args.max_samples:
                break

            name = image_list[n_seen]
            y = float(has_doc[i])
            s = float(score[i])

            if y >= 0.5:
                n_pos += 1
                iou = float(compute_polygon_iou(coords_pred[i], coords_gt[i]))
                err_px, _ = compute_corner_error(coords_pred[i], coords_gt[i], img_size=img_size)

                pred_is_pos = s >= args.score_threshold
                if args.include_fn and not pred_is_pos:
                    hard_fn.append((name, s))

                hard_pos.append((iou, name, err_px, s))
            else:
                n_neg += 1
                pred_is_pos = s >= args.score_threshold
                if args.include_fp and pred_is_pos:
                    hard_fp.append((name, s))

            n_seen += 1

        if args.max_samples and n_seen >= args.max_samples:
            break

    # Select positives
    selected_pos = []
    if args.top_k and args.top_k > 0:
        hard_pos_sorted = sorted(hard_pos, key=lambda t: t[0])  # IoU asc
        selected_pos = hard_pos_sorted[: args.top_k]
    else:
        for iou, name, err_px, s in hard_pos:
            if iou < args.iou_threshold:
                selected_pos.append((iou, name, err_px, s))
                continue
            if args.corner_err_threshold_px is not None and err_px > args.corner_err_threshold_px:
                selected_pos.append((iou, name, err_px, s))

    # Merge and write unique names (preserve a stable order)
    selected_names = []
    seen = set()

    def add(name: str):
        if name in seen:
            return
        seen.add(name)
        selected_names.append(name)

    for _, name, _, _ in sorted(selected_pos, key=lambda t: t[0]):
        add(name)
    for name, _ in sorted(hard_fp, key=lambda t: t[1], reverse=True):
        add(name)
    for name, _ in sorted(hard_fn, key=lambda t: t[1]):
        add(name)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for name in selected_names:
            f.write(f"{name}\n")

    pos_ious = np.array([t[0] for t in hard_pos], dtype=np.float32)
    if pos_ious.size:
        print("\nPositives IoU stats:")
        for q in (1, 5, 10, 25, 50, 75):
            print(f"  p{q:02d}: {np.percentile(pos_ious, q):.4f}")
        print(f"  mean: {pos_ious.mean():.4f}")

    print("\nSelection:")
    print(f"  seen: {n_seen} (pos={n_pos}, neg={n_neg})")
    print(f"  selected positives: {len(selected_pos)}")
    print(f"  selected FP negatives: {len(hard_fp)}")
    print(f"  selected FN positives: {len(hard_fn)}")
    print(f"  output: {out_path} ({len(selected_names)} names)")


if __name__ == "__main__":
    # Silence TF excessive logs for CLI usage
    tf.get_logger().setLevel("ERROR")
    main()

