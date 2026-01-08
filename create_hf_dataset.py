#!/usr/bin/env python3
"""
Create a Hugging Face dataset in Parquet format from doc-scanner-dataset-rev-new.

This script creates a dataset with:
- Images embedded as bytes (for HF Dataset Viewer compatibility)
- Corner coordinates (normalized 0-1)
- Split information (train/val/test)

Output structure:
    hf_dataset/
    ├── train/
    │   └── data-00000-of-NNNNN.parquet
    ├── val/
    │   └── data-00000-of-NNNNN.parquet
    ├── test/
    │   └── data-00000-of-NNNNN.parquet
    └── dataset_info.json
"""

import argparse
import json
from pathlib import Path
from PIL import Image
import io
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def load_split_files(split_file: Path) -> list[str]:
    """Load list of image filenames from split file."""
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def parse_label_file(label_path: Path) -> dict | None:
    """Parse YOLO 4-corner label file.

    Format: class x1 y1 x2 y2 x3 y3 x4 y4
    Returns dict with corner coordinates or None for negative images.
    """
    if not label_path.exists():
        return None

    content = label_path.read_text().strip()
    if not content:
        # Empty file = negative image (no document)
        return None

    parts = content.split()
    if len(parts) < 9:
        return None

    coords = [float(x) for x in parts[1:9]]

    return {
        "corner_tl_x": coords[0],
        "corner_tl_y": coords[1],
        "corner_tr_x": coords[2],
        "corner_tr_y": coords[3],
        "corner_br_x": coords[4],
        "corner_br_y": coords[5],
        "corner_bl_x": coords[6],
        "corner_bl_y": coords[7],
    }


def image_to_bytes(image_path: Path) -> bytes:
    """Load image and convert to JPEG bytes."""
    with Image.open(image_path) as img:
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if img.mode != "RGB":
            img = img.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()


def create_parquet_for_split(
    split_name: str,
    split_files: list[str],
    images_dir: Path,
    images_negative_dir: Path,
    labels_dir: Path,
    labels_negative_dir: Path,
    output_dir: Path,
    rows_per_file: int = 1000,
):
    """Create Parquet files for a split."""

    output_split_dir = output_dir / split_name
    output_split_dir.mkdir(parents=True, exist_ok=True)

    # Prepare schema
    schema = pa.schema([
        ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("filename", pa.string()),
        ("is_negative", pa.bool_()),
        ("corner_tl_x", pa.float32()),
        ("corner_tl_y", pa.float32()),
        ("corner_tr_x", pa.float32()),
        ("corner_tr_y", pa.float32()),
        ("corner_br_x", pa.float32()),
        ("corner_br_y", pa.float32()),
        ("corner_bl_x", pa.float32()),
        ("corner_bl_y", pa.float32()),
    ])

    rows = []
    file_idx = 0

    for filename in tqdm(split_files, desc=f"Processing {split_name}"):
        # Check if it's a negative image
        is_negative = filename.startswith("negative_")

        if is_negative:
            image_path = images_negative_dir / filename
            label_path = labels_negative_dir / f"{Path(filename).stem}.txt"
        else:
            image_path = images_dir / filename
            label_path = labels_dir / f"{Path(filename).stem}.txt"

        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        # Load image bytes
        try:
            image_bytes = image_to_bytes(image_path)
        except Exception as e:
            print(f"Warning: Failed to load image {image_path}: {e}")
            continue

        # Parse label
        corners = parse_label_file(label_path)

        row = {
            "image": {"bytes": image_bytes, "path": filename},
            "filename": filename,
            "is_negative": corners is None,
            "corner_tl_x": corners["corner_tl_x"] if corners else None,
            "corner_tl_y": corners["corner_tl_y"] if corners else None,
            "corner_tr_x": corners["corner_tr_x"] if corners else None,
            "corner_tr_y": corners["corner_tr_y"] if corners else None,
            "corner_br_x": corners["corner_br_x"] if corners else None,
            "corner_br_y": corners["corner_br_y"] if corners else None,
            "corner_bl_x": corners["corner_bl_x"] if corners else None,
            "corner_bl_y": corners["corner_bl_y"] if corners else None,
        }
        rows.append(row)

        # Write batch to Parquet
        if len(rows) >= rows_per_file:
            table = pa.Table.from_pylist(rows, schema=schema)
            output_file = output_split_dir / f"data-{file_idx:05d}.parquet"
            pq.write_table(table, output_file, compression="snappy")
            file_idx += 1
            rows = []

    # Write remaining rows
    if rows:
        table = pa.Table.from_pylist(rows, schema=schema)
        output_file = output_split_dir / f"data-{file_idx:05d}.parquet"
        pq.write_table(table, output_file, compression="snappy")

    return file_idx + 1 if rows or file_idx > 0 else 0


def create_dataset_info(output_dir: Path, train_count: int, val_count: int, test_count: int):
    """Create dataset_info.json for Hugging Face."""

    info = {
        "description": "Document Corner Detection Dataset - Best generalist training data from doc-scanner-dataset-rev-new with train_clean_iter3_plus_hard_full, val_clean_iter3, and test splits.",
        "citation": "",
        "homepage": "https://huggingface.co/datasets/mapo80/DocCornerDataset",
        "license": "MIT",
        "features": {
            "image": {"_type": "Image"},
            "filename": {"dtype": "string", "_type": "Value"},
            "is_negative": {"dtype": "bool", "_type": "Value"},
            "corner_tl_x": {"dtype": "float32", "_type": "Value"},
            "corner_tl_y": {"dtype": "float32", "_type": "Value"},
            "corner_tr_x": {"dtype": "float32", "_type": "Value"},
            "corner_tr_y": {"dtype": "float32", "_type": "Value"},
            "corner_br_x": {"dtype": "float32", "_type": "Value"},
            "corner_br_y": {"dtype": "float32", "_type": "Value"},
            "corner_bl_x": {"dtype": "float32", "_type": "Value"},
            "corner_bl_y": {"dtype": "float32", "_type": "Value"},
        },
        "splits": {
            "train": {"name": "train", "num_examples": train_count},
            "validation": {"name": "validation", "num_examples": val_count},
            "test": {"name": "test", "num_examples": test_count},
        },
        "dataset_name": "DocCornerDataset-BestGeneralist",
        "config_name": "default",
    }

    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)


def create_readme(output_dir: Path, train_count: int, val_count: int, test_count: int):
    """Create README.md for Hugging Face dataset."""

    readme = f"""---
dataset_info:
  features:
    - name: image
      dtype: image
    - name: filename
      dtype: string
    - name: is_negative
      dtype: bool
    - name: corner_tl_x
      dtype: float32
    - name: corner_tl_y
      dtype: float32
    - name: corner_tr_x
      dtype: float32
    - name: corner_tr_y
      dtype: float32
    - name: corner_br_x
      dtype: float32
    - name: corner_br_y
      dtype: float32
    - name: corner_bl_x
      dtype: float32
    - name: corner_bl_y
      dtype: float32
  splits:
    - name: train
      num_examples: {train_count}
    - name: validation
      num_examples: {val_count}
    - name: test
      num_examples: {test_count}
configs:
  - config_name: default
    data_files:
      - split: train
        path: train/*.parquet
      - split: validation
        path: val/*.parquet
      - split: test
        path: test/*.parquet
license: mit
task_categories:
  - image-segmentation
  - keypoint-detection
tags:
  - document-detection
  - corner-detection
  - perspective-correction
---

# DocCornerDataset - Best Generalist

Document corner detection dataset for training models to detect the four corners of documents in images.

## Dataset Description

This dataset contains images with document corner annotations, optimized for training robust document detection models. It uses the best-performing splits (`train_clean_iter3_plus_hard_full` / `val_clean_iter3` / `test`) from the iterative dataset cleaning process.

### Splits

| Split | Images | Description |
|-------|--------|-------------|
| train | {train_count:,} | Training set (cleaned iter3 + hard negatives) |
| validation | {val_count:,} | Validation set (cleaned iter3) |
| test | {test_count:,} | Held-out test set (no overlap with train/val) |

### Features

- **image**: The document image (JPEG)
- **filename**: Original filename
- **is_negative**: Boolean indicating if the image contains no document (negative sample)
- **corner_tl_x/y**: Top-left corner coordinates (normalized 0-1)
- **corner_tr_x/y**: Top-right corner coordinates (normalized 0-1)
- **corner_br_x/y**: Bottom-right corner coordinates (normalized 0-1)
- **corner_bl_x/y**: Bottom-left corner coordinates (normalized 0-1)

### Corner Order

Corners are ordered clockwise starting from top-left:
1. Top-left (tl)
2. Top-right (tr)
3. Bottom-right (br)
4. Bottom-left (bl)

### Negative Samples

Images with `is_negative=True` do not contain documents. All corner coordinates will be `null` for these samples.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("mapo80/DocCornerDataset-BestGeneralist")

# Access training data
for sample in dataset["train"]:
    image = sample["image"]
    if not sample["is_negative"]:
        corners = [
            (sample["corner_tl_x"], sample["corner_tl_y"]),
            (sample["corner_tr_x"], sample["corner_tr_y"]),
            (sample["corner_br_x"], sample["corner_br_y"]),
            (sample["corner_bl_x"], sample["corner_bl_y"]),
        ]
        # corners are normalized (0-1), multiply by image dimensions for pixel coords

# Evaluate on test set
for sample in dataset["test"]:
    # ...
```

## License

MIT License
"""

    with open(output_dir / "README.md", "w") as f:
        f.write(readme)


def main():
    parser = argparse.ArgumentParser(description="Create HF dataset from doc-scanner-dataset-rev-new")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new"),
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train_clean_iter3_plus_hard_full",
        help="Training split filename (without .txt)",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="val_clean_iter3",
        help="Validation split filename (without .txt)",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="test",
        help="Test split filename (without .txt)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./hf_dataset"),
        help="Output directory for Parquet files",
    )
    parser.add_argument(
        "--rows_per_file",
        type=int,
        default=1000,
        help="Number of rows per Parquet file",
    )
    args = parser.parse_args()

    # Paths
    images_dir = args.data_root / "images"
    images_negative_dir = args.data_root / "images-negative"
    labels_dir = args.data_root / "labels"
    labels_negative_dir = args.data_root / "labels-negative"
    train_split_file = args.data_root / f"{args.train_split}.txt"
    val_split_file = args.data_root / f"{args.val_split}.txt"
    test_split_file = args.data_root / f"{args.test_split}.txt"

    # Validate paths
    for path in [images_dir, labels_dir, train_split_file, val_split_file, test_split_file]:
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    print(f"Loading train split from {train_split_file}")
    train_files = load_split_files(train_split_file)
    print(f"  Found {len(train_files)} training images")

    print(f"Loading val split from {val_split_file}")
    val_files = load_split_files(val_split_file)
    print(f"  Found {len(val_files)} validation images")

    print(f"Loading test split from {test_split_file}")
    test_files = load_split_files(test_split_file)
    print(f"  Found {len(test_files)} test images")

    # Create Parquet files
    print("\nCreating training Parquet files...")
    train_parquet_count = create_parquet_for_split(
        "train",
        train_files,
        images_dir,
        images_negative_dir,
        labels_dir,
        labels_negative_dir,
        args.output_dir,
        args.rows_per_file,
    )
    print(f"  Created {train_parquet_count} Parquet files for training")

    print("\nCreating validation Parquet files...")
    val_parquet_count = create_parquet_for_split(
        "val",
        val_files,
        images_dir,
        images_negative_dir,
        labels_dir,
        labels_negative_dir,
        args.output_dir,
        args.rows_per_file,
    )
    print(f"  Created {val_parquet_count} Parquet files for validation")

    print("\nCreating test Parquet files...")
    test_parquet_count = create_parquet_for_split(
        "test",
        test_files,
        images_dir,
        images_negative_dir,
        labels_dir,
        labels_negative_dir,
        args.output_dir,
        args.rows_per_file,
    )
    print(f"  Created {test_parquet_count} Parquet files for test")

    # Create metadata files
    print("\nCreating dataset metadata...")
    create_dataset_info(args.output_dir, len(train_files), len(val_files), len(test_files))
    create_readme(args.output_dir, len(train_files), len(val_files), len(test_files))

    print(f"\nDataset created successfully in {args.output_dir}")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val: {len(val_files)} images")
    print(f"  Test: {len(test_files)} images")
    print(f"\nTo upload to Hugging Face:")
    print(f"  huggingface-cli upload mapo80/DocCornerDataset-BestGeneralist {args.output_dir} --repo-type dataset")


if __name__ == "__main__":
    main()
