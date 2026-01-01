"""
Generate model predictions for all positive images in the dataset.

Creates new label files with class 1 (purple bbox) containing model predictions,
while preserving original GT labels with class 0 (green bbox).

Usage:
    python generate_predictions.py \
        --checkpoint checkpoints/mobilenetv2_224_iou98 \
        --dataset /path/to/doc-scanner-dataset-reviewed \
        --output_dir labels-predicted
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import tensorflow as tf

from tensorflow import keras
from model import load_inference_model

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
    # Resize
    img = Image.fromarray(image)
    img = img.resize((img_size, img_size), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0

    # ImageNet normalization
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    return img


def get_all_images(dataset_path: Path, exclude_negative: bool = True) -> List[Path]:
    """Get all positive image paths from dataset."""
    images_dir = dataset_path / "images"

    image_files = []
    for ext in IMAGE_EXTS:
        image_files.extend(images_dir.glob(f"*{ext}"))

    return sorted(image_files)


def coords_to_yolo_obb(coords: np.ndarray) -> str:
    """
    Convert normalized corner coordinates to YOLO OBB format.

    coords: [x0, y0, x1, y1, x2, y2, x3, y3] - normalized [0,1]
    Returns: "class x0 y0 x1 y1 x2 y2 x3 y3" (TL, TR, BR, BL order)
    """
    # coords are already in TL, TR, BR, BL order from model output
    return f"1 {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f} {coords[4]:.6f} {coords[5]:.6f} {coords[6]:.6f} {coords[7]:.6f}"


def main():
    parser = argparse.ArgumentParser(description="Generate model predictions for dataset")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset root")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--score_threshold", type=float, default=0.5,
                        help="Minimum score threshold for predictions")
    parser.add_argument("--append_to_labels", action="store_true",
                        help="Append predictions to existing label files instead of creating new ones")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    dataset_path = Path(args.dataset)
    labels_dir = dataset_path / "labels"

    # Load config
    config = load_config(checkpoint_path)
    img_size = config.get("img_size", 224)
    print(f"Model config: img_size={img_size}")

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    weights_path = checkpoint_path / "best_model.weights.h5"
    model = load_inference_model(
        weights_path,
        backbone=config.get("backbone", "mobilenetv2"),
        alpha=config.get("alpha", 0.35),
        fpn_ch=config.get("fpn_ch", 32),
        simcc_ch=config.get("simcc_ch", 96),
        img_size=img_size,
        num_bins=config.get("num_bins", img_size),
    )
    print("Model loaded.")

    # Get all positive images
    print("Scanning for images...")
    image_files = get_all_images(dataset_path)
    print(f"Found {len(image_files)} positive images")

    # Process in batches
    num_processed = 0
    num_skipped = 0
    num_low_score = 0

    batch_images = []
    batch_paths = []

    for img_path in tqdm(image_files, desc="Processing images"):
        # Load and preprocess image
        try:
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            preprocessed = preprocess_image(img_array, img_size)
            batch_images.append(preprocessed)
            batch_paths.append(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            num_skipped += 1
            continue

        # Process batch
        if len(batch_images) >= args.batch_size:
            batch_input = np.stack(batch_images, axis=0)
            predictions = model.predict(batch_input, verbose=0)

            for i, pred in enumerate(predictions):
                coords = pred[:8]  # x0,y0,x1,y1,x2,y2,x3,y3
                score = pred[8]

                img_path = batch_paths[i]
                label_path = labels_dir / (img_path.stem + ".txt")

                if score < args.score_threshold:
                    num_low_score += 1
                    continue

                # Create prediction line (class 1 = purple)
                pred_line = coords_to_yolo_obb(coords)

                if args.append_to_labels:
                    # Append to existing label file
                    if label_path.exists():
                        with open(label_path, "a") as f:
                            f.write(f"\n{pred_line}")
                    else:
                        with open(label_path, "w") as f:
                            f.write(pred_line)
                else:
                    # Read existing GT and append prediction
                    lines = []
                    if label_path.exists():
                        with open(label_path, "r") as f:
                            lines = [l.strip() for l in f.readlines() if l.strip()]

                    # Filter out any existing class 1 predictions
                    lines = [l for l in lines if not l.startswith("1 ")]

                    # Append new prediction
                    lines.append(pred_line)

                    # Write back
                    with open(label_path, "w") as f:
                        f.write("\n".join(lines))

                num_processed += 1

            batch_images = []
            batch_paths = []

    # Process remaining images
    if batch_images:
        batch_input = np.stack(batch_images, axis=0)
        predictions = model.predict(batch_input, verbose=0)

        for i, pred in enumerate(predictions):
            coords = pred[:8]
            score = pred[8]

            img_path = batch_paths[i]
            label_path = labels_dir / (img_path.stem + ".txt")

            if score < args.score_threshold:
                num_low_score += 1
                continue

            pred_line = coords_to_yolo_obb(coords)

            if args.append_to_labels:
                if label_path.exists():
                    with open(label_path, "a") as f:
                        f.write(f"\n{pred_line}")
                else:
                    with open(label_path, "w") as f:
                        f.write(pred_line)
            else:
                lines = []
                if label_path.exists():
                    with open(label_path, "r") as f:
                        lines = [l.strip() for l in f.readlines() if l.strip()]

                lines = [l for l in lines if not l.startswith("1 ")]
                lines.append(pred_line)

                with open(label_path, "w") as f:
                    f.write("\n".join(lines))

            num_processed += 1

    print(f"\nDone!")
    print(f"  Processed: {num_processed}")
    print(f"  Skipped (errors): {num_skipped}")
    print(f"  Low score (<{args.score_threshold}): {num_low_score}")


if __name__ == "__main__":
    main()
