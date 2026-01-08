#!/usr/bin/env python3
"""Create collage images for HuggingFace dataset README."""

import random
from pathlib import Path
from PIL import Image, ImageDraw
import pyarrow.parquet as pq


def load_samples_from_parquet(parquet_dir: Path, num_samples: int = 9) -> list:
    """Load random samples from parquet files."""
    parquet_files = sorted(parquet_dir.glob("*.parquet"))

    # Read all filenames first to sample randomly
    all_rows = []
    for pf in parquet_files:
        table = pq.read_table(pf)
        for i in range(len(table)):
            row = {
                "image_bytes": table["image"][i].as_py()["bytes"],
                "filename": table["filename"][i].as_py(),
                "is_negative": table["is_negative"][i].as_py(),
                "corners": None if table["is_negative"][i].as_py() else [
                    (table["corner_tl_x"][i].as_py(), table["corner_tl_y"][i].as_py()),
                    (table["corner_tr_x"][i].as_py(), table["corner_tr_y"][i].as_py()),
                    (table["corner_br_x"][i].as_py(), table["corner_br_y"][i].as_py()),
                    (table["corner_bl_x"][i].as_py(), table["corner_bl_y"][i].as_py()),
                ]
            }
            all_rows.append(row)

    # Sample randomly, preferring positive samples
    positive_rows = [r for r in all_rows if not r["is_negative"]]
    negative_rows = [r for r in all_rows if r["is_negative"]]

    # Take mostly positive samples (8 positive, 1 negative if available)
    random.seed(42)
    samples = random.sample(positive_rows, min(8, len(positive_rows)))
    if negative_rows:
        samples.append(random.choice(negative_rows))

    # Shuffle
    random.shuffle(samples)
    return samples[:num_samples]


def draw_corners(img: Image.Image, corners: list, color=(0, 255, 0), thickness: int = 3) -> Image.Image:
    """Draw corner polygon on image."""
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Convert normalized coords to pixels
    points = [(int(c[0] * w), int(c[1] * h)) for c in corners]

    # Draw polygon
    for i in range(4):
        start = points[i]
        end = points[(i + 1) % 4]
        draw.line([start, end], fill=color, width=thickness)

    # Draw corner circles
    for p in points:
        r = 6
        draw.ellipse([p[0]-r, p[1]-r, p[0]+r, p[1]+r], fill=color)

    return img


def create_collage(samples: list, output_path: Path, grid_size: int = 3, cell_size: int = 300):
    """Create a grid collage from samples."""
    collage_size = grid_size * cell_size
    collage = Image.new("RGB", (collage_size, collage_size), (255, 255, 255))

    for idx, sample in enumerate(samples[:grid_size * grid_size]):
        row = idx // grid_size
        col = idx % grid_size

        # Load image from bytes
        import io
        img = Image.open(io.BytesIO(sample["image_bytes"]))
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Draw corners if positive
        if sample["corners"]:
            img = draw_corners(img, sample["corners"])

        # Resize to cell size
        img = img.resize((cell_size, cell_size), Image.Resampling.LANCZOS)

        # Paste into collage
        x = col * cell_size
        y = row * cell_size
        collage.paste(img, (x, y))

    collage.save(output_path, quality=90)
    print(f"Saved collage to {output_path}")


def main():
    hf_dataset_dir = Path("./hf_dataset")

    # Create collages directory
    collages_dir = hf_dataset_dir / "collages"
    collages_dir.mkdir(exist_ok=True)

    for split in ["train", "val", "test"]:
        print(f"\nProcessing {split} split...")
        split_dir = hf_dataset_dir / split

        if not split_dir.exists():
            print(f"  Split directory not found: {split_dir}")
            continue

        samples = load_samples_from_parquet(split_dir, num_samples=9)
        print(f"  Loaded {len(samples)} samples")

        output_path = collages_dir / f"{split}_collage.jpg"
        create_collage(samples, output_path)


if __name__ == "__main__":
    main()
