"""
Find worst predictions (outliers) based on corner error >= threshold pixels.

Usage:
    python find_outliers.py \
        --checkpoint checkpoints/mobilenetv2_224_clean \
        --train_file /path/to/train.txt \
        --val_file /path/to/val.txt \
        --dataset /path/to/dataset \
        --output_dir outliers_output \
        --threshold 10
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import tensorflow as tf
from model import load_inference_model
from metrics import compute_corner_error, compute_polygon_iou

# Constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp", ".WEBP"]


def load_config(checkpoint_path: Path) -> dict:
    """Load model config from checkpoint directory."""
    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {"img_size": 224, "num_bins": 224}


def preprocess_image(image: np.ndarray, img_size: int) -> np.ndarray:
    """Preprocess image for model inference."""
    img = Image.fromarray(image)
    img = img.resize((img_size, img_size), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img


def load_split_file(split_path: str) -> List[str]:
    """Load image names from split file."""
    result = []
    with open(split_path, 'r') as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            # Remove directory prefixes if present
            if name.startswith("images-negative/"):
                name = name[len("images-negative/"):]
            elif name.startswith("images/"):
                name = name[len("images/"):]
            result.append(name)
    return result


def load_yolo_label(label_path: str) -> Tuple[np.ndarray, bool]:
    """Load coordinates from YOLO OBB format label file."""
    if not os.path.exists(label_path):
        return np.zeros(8, dtype=np.float32), False

    with open(label_path, 'r') as f:
        line = f.readline().strip()

    if not line:
        return np.zeros(8, dtype=np.float32), False

    parts = line.split()
    if len(parts) < 9:
        return np.zeros(8, dtype=np.float32), False

    coords = np.array([float(x) for x in parts[1:9]], dtype=np.float32)
    return coords, True


def find_image_path(image_name: str, images_dir: Path, negative_dir: Path) -> Path:
    """Find image path trying different directories and extensions."""
    # Try positive images directory
    for ext in IMAGE_EXTS:
        stem = Path(image_name).stem
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
        # Try exact name
        candidate = images_dir / image_name
        if candidate.exists():
            return candidate

    # Try negative images directory
    if negative_dir.exists():
        for ext in IMAGE_EXTS:
            stem = Path(image_name).stem
            candidate = negative_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
            candidate = negative_dir / image_name
            if candidate.exists():
                return candidate

    return images_dir / image_name


def draw_bbox_on_image(
    image: Image.Image,
    gt_coords: np.ndarray,
    pred_coords: np.ndarray,
    corner_error_px: float,
    iou: float,
    img_size: int = 224
) -> Image.Image:
    """
    Draw GT (green) and predicted (red) bounding boxes on image.

    Args:
        image: PIL Image
        gt_coords: [8] normalized GT coordinates
        pred_coords: [8] normalized predicted coordinates
        corner_error_px: Corner error in pixels
        iou: IoU value

    Returns:
        Image with bboxes drawn
    """
    # Resize image to model input size for consistent visualization
    img = image.resize((img_size, img_size), Image.BILINEAR)
    draw = ImageDraw.Draw(img)

    # Convert normalized coords to pixel coords
    gt_px = (gt_coords * img_size).reshape(4, 2)
    pred_px = (pred_coords * img_size).reshape(4, 2)

    # Draw GT bbox (green)
    gt_points = [(gt_px[i, 0], gt_px[i, 1]) for i in range(4)]
    gt_points.append(gt_points[0])  # Close polygon
    draw.line(gt_points, fill=(0, 255, 0), width=2)

    # Draw predicted bbox (red)
    pred_points = [(pred_px[i, 0], pred_px[i, 1]) for i in range(4)]
    pred_points.append(pred_points[0])  # Close polygon
    draw.line(pred_points, fill=(255, 0, 0), width=2)

    # Draw corner markers
    for i in range(4):
        # GT corner (green circle)
        x, y = gt_px[i]
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(0, 255, 0))

        # Predicted corner (red circle)
        x, y = pred_px[i]
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 0, 0))

    # Add text with metrics
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font = ImageFont.load_default()

    text = f"Err: {corner_error_px:.1f}px  IoU: {iou:.3f}"
    draw.rectangle([0, 0, img_size, 16], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255), font=font)

    return img


def process_split(
    model,
    split_file: str,
    dataset_path: Path,
    img_size: int,
    batch_size: int,
    threshold_px: float,
) -> List[Dict]:
    """
    Process a split file and find outliers.

    Returns:
        List of dicts with outlier info, sorted by corner error (descending)
    """
    images_dir = dataset_path / "images"
    negative_dir = dataset_path / "images-negative"
    labels_dir = dataset_path / "labels"

    # Load split
    image_names = load_split_file(split_file)
    print(f"  Found {len(image_names)} images in split")

    # Filter to positive samples only (negative samples have no GT)
    positive_samples = []
    for name in image_names:
        if name.startswith("negative_") or name == "negative":
            continue
        label_path = labels_dir / f"{Path(name).stem}.txt"
        gt_coords, has_doc = load_yolo_label(str(label_path))
        if has_doc:
            positive_samples.append((name, gt_coords))

    print(f"  Found {len(positive_samples)} positive samples with GT")

    outliers = []
    batch_images = []
    batch_info = []  # (name, gt_coords, img_path)

    for name, gt_coords in tqdm(positive_samples, desc="  Processing"):
        img_path = find_image_path(name, images_dir, negative_dir)

        if not img_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            preprocessed = preprocess_image(img_array, img_size)
            batch_images.append(preprocessed)
            batch_info.append((name, gt_coords, img_path))
        except Exception as e:
            print(f"  Error loading {name}: {e}")
            continue

        # Process batch
        if len(batch_images) >= batch_size:
            batch_input = np.stack(batch_images, axis=0)
            # Model returns [coords, score_logit] as list
            coords_batch, scores_batch = model.predict(batch_input, verbose=0)

            for i in range(len(batch_info)):
                pred_coords = coords_batch[i]  # [8]
                score = float(tf.nn.sigmoid(scores_batch[i]).numpy())
                name_i, gt_coords_i, img_path_i = batch_info[i]

                # Compute metrics
                mean_err, per_corner = compute_corner_error(pred_coords, gt_coords_i, img_size)
                iou = compute_polygon_iou(pred_coords, gt_coords_i)

                if mean_err >= threshold_px:
                    outliers.append({
                        "name": name_i,
                        "img_path": str(img_path_i),
                        "corner_error_px": mean_err,
                        "per_corner_error_px": per_corner.tolist(),
                        "iou": iou,
                        "score": score,
                        "gt_coords": gt_coords_i.tolist(),
                        "pred_coords": pred_coords.tolist(),
                    })

            batch_images = []
            batch_info = []

    # Process remaining
    if batch_images:
        batch_input = np.stack(batch_images, axis=0)
        coords_batch, scores_batch = model.predict(batch_input, verbose=0)

        for i in range(len(batch_info)):
            pred_coords = coords_batch[i]
            score = float(tf.nn.sigmoid(scores_batch[i]).numpy())
            name_i, gt_coords_i, img_path_i = batch_info[i]

            mean_err, per_corner = compute_corner_error(pred_coords, gt_coords_i, img_size)
            iou = compute_polygon_iou(pred_coords, gt_coords_i)

            if mean_err >= threshold_px:
                outliers.append({
                    "name": name_i,
                    "img_path": str(img_path_i),
                    "corner_error_px": mean_err,
                    "per_corner_error_px": per_corner.tolist(),
                    "iou": iou,
                    "score": score,
                    "gt_coords": gt_coords_i.tolist(),
                    "pred_coords": pred_coords.tolist(),
                })

    # Sort by corner error (worst first)
    outliers.sort(key=lambda x: x["corner_error_px"], reverse=True)

    return outliers


def main():
    parser = argparse.ArgumentParser(description="Find worst predictions (outliers)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to train.txt split file")
    parser.add_argument("--val_file", type=str, required=True,
                        help="Path to val.txt split file")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset root (containing images/, labels/)")
    parser.add_argument("--output_dir", type=str, default="outliers_output",
                        help="Output directory for results")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="Corner error threshold in pixels (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--max_images", type=int, default=100,
                        help="Maximum number of visualization images to save")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_train_dir = output_dir / "visualizations" / "train"
    vis_val_dir = output_dir / "visualizations" / "val"
    vis_train_dir.mkdir(parents=True, exist_ok=True)
    vis_val_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(checkpoint_path)
    img_size = config.get("img_size", 224)
    print(f"Model config: img_size={img_size}")

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    weights_path = checkpoint_path / "best_model.weights.h5"
    model = load_inference_model(
        str(weights_path),
        backbone=config.get("backbone", "mobilenetv2"),
        alpha=config.get("alpha", 0.35),
        fpn_ch=config.get("fpn_ch", 32),
        simcc_ch=config.get("simcc_ch", 96),
        img_size=img_size,
        num_bins=config.get("num_bins", img_size),
    )
    print("Model loaded.")

    # Process train split
    print(f"\nProcessing train split: {args.train_file}")
    train_outliers = process_split(
        model, args.train_file, dataset_path, img_size,
        args.batch_size, args.threshold
    )
    print(f"  Found {len(train_outliers)} outliers (>= {args.threshold}px)")

    # Process val split
    print(f"\nProcessing val split: {args.val_file}")
    val_outliers = process_split(
        model, args.val_file, dataset_path, img_size,
        args.batch_size, args.threshold
    )
    print(f"  Found {len(val_outliers)} outliers (>= {args.threshold}px)")

    # Combine all outliers
    all_outliers = train_outliers + val_outliers
    all_outliers.sort(key=lambda x: x["corner_error_px"], reverse=True)

    # Write outliers.txt (just image names)
    outliers_file = output_dir / "outliers.txt"
    with open(outliers_file, "w") as f:
        for outlier in all_outliers:
            f.write(f"{outlier['name']}\n")
    print(f"\nWrote {len(all_outliers)} outliers to {outliers_file}")

    # Write detailed outliers JSON
    outliers_json = output_dir / "outliers_detailed.json"
    with open(outliers_json, "w") as f:
        json.dump({
            "threshold_px": args.threshold,
            "num_train_outliers": len(train_outliers),
            "num_val_outliers": len(val_outliers),
            "train_outliers": train_outliers,
            "val_outliers": val_outliers,
        }, f, indent=2)
    print(f"Wrote detailed info to {outliers_json}")

    # Generate visualization images
    print(f"\nGenerating visualization images (max {args.max_images} per split)...")

    # Train visualizations
    for i, outlier in enumerate(train_outliers[:args.max_images]):
        img_path = Path(outlier["img_path"])
        try:
            img = Image.open(img_path).convert("RGB")
            gt_coords = np.array(outlier["gt_coords"])
            pred_coords = np.array(outlier["pred_coords"])

            vis_img = draw_bbox_on_image(
                img, gt_coords, pred_coords,
                outlier["corner_error_px"], outlier["iou"], img_size
            )

            # Save with error in filename for easy sorting
            error_str = f"{outlier['corner_error_px']:.1f}".replace(".", "_")
            vis_path = vis_train_dir / f"{error_str}px_{Path(outlier['name']).stem}.png"
            vis_img.save(vis_path)
        except Exception as e:
            print(f"  Error creating visualization for {outlier['name']}: {e}")

    print(f"  Saved {min(len(train_outliers), args.max_images)} train visualizations to {vis_train_dir}")

    # Val visualizations
    for i, outlier in enumerate(val_outliers[:args.max_images]):
        img_path = Path(outlier["img_path"])
        try:
            img = Image.open(img_path).convert("RGB")
            gt_coords = np.array(outlier["gt_coords"])
            pred_coords = np.array(outlier["pred_coords"])

            vis_img = draw_bbox_on_image(
                img, gt_coords, pred_coords,
                outlier["corner_error_px"], outlier["iou"], img_size
            )

            error_str = f"{outlier['corner_error_px']:.1f}".replace(".", "_")
            vis_path = vis_val_dir / f"{error_str}px_{Path(outlier['name']).stem}.png"
            vis_img.save(vis_path)
        except Exception as e:
            print(f"  Error creating visualization for {outlier['name']}: {e}")

    print(f"  Saved {min(len(val_outliers), args.max_images)} val visualizations to {vis_val_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Threshold: >= {args.threshold}px corner error")
    print(f"Train outliers: {len(train_outliers)}")
    print(f"Val outliers: {len(val_outliers)}")
    print(f"Total outliers: {len(all_outliers)}")

    if all_outliers:
        print(f"\nWorst predictions:")
        for outlier in all_outliers[:10]:
            print(f"  {outlier['name']}: {outlier['corner_error_px']:.1f}px, IoU={outlier['iou']:.3f}")

    print(f"\nOutput files:")
    print(f"  {outliers_file}")
    print(f"  {outliers_json}")
    print(f"  {vis_train_dir}/")
    print(f"  {vis_val_dir}/")


if __name__ == "__main__":
    main()
