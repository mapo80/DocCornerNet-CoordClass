"""
Visualization script for DocCornerNetV3 augmentation pipeline.

Creates visualization samples showing:
- Original images vs augmented images
- Positive samples with document corners
- Negative samples (no document)
- Outlier samples with specific augmentation

Outputs images to visualizations/ folder.
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dataset import DocCornerDataset, DEFAULT_AUG_CONFIG


def draw_corners(image: Image.Image, coords: np.ndarray, color: tuple = (255, 0, 0), width: int = 2) -> Image.Image:
    """
    Draw document corners on image.

    Args:
        image: PIL Image
        coords: [8] normalized coordinates (x0,y0,x1,y1,x2,y2,x3,y3)
        color: RGB color tuple
        width: Line width

    Returns:
        Image with corners drawn
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Convert normalized coords to pixels
    points = []
    for i in range(0, 8, 2):
        x = coords[i] * w
        y = coords[i + 1] * h
        points.append((x, y))

    # Draw quadrilateral
    if len(points) == 4:
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            draw.line([p1, p2], fill=color, width=width)

        # Draw corner circles
        for i, p in enumerate(points):
            r = 4
            draw.ellipse([p[0]-r, p[1]-r, p[0]+r, p[1]+r], fill=color)
            # Label corners
            labels = ["TL", "TR", "BR", "BL"]
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
            except:
                font = ImageFont.load_default()
            draw.text((p[0]+5, p[1]-15), labels[i], fill=color, font=font)

    return img


def create_comparison_image(
    original: Image.Image,
    augmented: Image.Image,
    original_coords: np.ndarray,
    augmented_coords: np.ndarray,
    has_doc: bool,
    is_outlier: bool,
    image_name: str,
    img_size: int = 224,
) -> Image.Image:
    """
    Create side-by-side comparison image.

    Layout:
    [Original with corners] | [Augmented with corners]
    [Label info]
    """
    # Create canvas
    margin = 10
    label_height = 40
    canvas_width = img_size * 2 + margin * 3
    canvas_height = img_size + margin * 2 + label_height

    canvas = Image.new("RGB", (canvas_width, canvas_height), (240, 240, 240))

    # Draw original with corners
    if has_doc:
        original_vis = draw_corners(original, original_coords, color=(0, 255, 0), width=2)
    else:
        original_vis = original

    # Draw augmented with corners
    if has_doc:
        augmented_vis = draw_corners(augmented, augmented_coords, color=(255, 0, 0), width=2)
    else:
        augmented_vis = augmented

    # Paste images
    canvas.paste(original_vis, (margin, margin))
    canvas.paste(augmented_vis, (margin * 2 + img_size, margin))

    # Draw labels
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except:
        font = ImageFont.load_default()
        font_small = font

    # Type label
    if not has_doc:
        type_label = "NEGATIVE"
        type_color = (128, 128, 128)
    elif is_outlier:
        type_label = "OUTLIER"
        type_color = (255, 128, 0)
    else:
        type_label = "POSITIVE"
        type_color = (0, 128, 0)

    y_text = img_size + margin + 5

    # Original label
    draw.text((margin, y_text), "Original (green)", fill=(0, 128, 0), font=font_small)

    # Augmented label
    draw.text((margin * 2 + img_size, y_text), "Augmented (red)", fill=(200, 0, 0), font=font_small)

    # Type and name
    draw.text((margin, y_text + 18), f"[{type_label}] {image_name[:40]}", fill=type_color, font=font_small)

    return canvas


def visualize_dataset(
    data_root: str,
    output_dir: str,
    num_positive: int = 5,
    num_negative: int = 5,
    num_outliers: int = 5,
    num_augmentations: int = 3,
    img_size: int = 224,
    outlier_list: str = None,
    augment_config: dict = None,
    augment_config_outlier: dict = None,
):
    """
    Generate visualization samples.

    Args:
        data_root: Path to dataset root
        output_dir: Output directory for visualizations
        num_positive: Number of positive samples to visualize
        num_negative: Number of negative samples to visualize
        num_outliers: Number of outlier samples to visualize
        num_augmentations: Number of augmentation variants per sample
        img_size: Image size
        outlier_list: Path to outlier list file
        augment_config: Augmentation config for normal samples
        augment_config_outlier: Augmentation config for outliers
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Default augmentation configs
    if augment_config is None:
        augment_config = DEFAULT_AUG_CONFIG.copy()

    if augment_config_outlier is None:
        augment_config_outlier = DEFAULT_AUG_CONFIG.copy()
        # Outliers might need stronger augmentation
        augment_config_outlier["rotation_degrees"] = 8
        augment_config_outlier["scale_range"] = (0.85, 1.0)
        augment_config_outlier["brightness"] = 0.25
        augment_config_outlier["contrast"] = 0.25

    # Create dataset with augmentation enabled
    dataset = DocCornerDataset(
        data_root=data_root,
        split="train",
        img_size=img_size,
        augment=True,
        augment_config=augment_config,
        augment_config_outlier=augment_config_outlier,
        outlier_list=outlier_list,
    )

    print(f"\nDataset loaded:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Positive samples: {len(dataset.positive_samples)}")
    print(f"  Negative samples: {len(dataset.negative_samples)}")
    print(f"  Outliers: {len(dataset.outlier_names)}")

    # Collect samples by type
    positive_indices = []
    negative_indices = []
    outlier_indices = []

    for idx, name in enumerate(dataset.image_list):
        if name.startswith("negative_"):
            negative_indices.append(idx)
        elif name in dataset.outlier_names:
            outlier_indices.append(idx)
        else:
            positive_indices.append(idx)

    print(f"\n  Positive indices: {len(positive_indices)}")
    print(f"  Negative indices: {len(negative_indices)}")
    print(f"  Outlier indices: {len(outlier_indices)}")

    # Generate visualizations
    visualizations = []

    # Positive samples
    print(f"\nGenerating positive sample visualizations...")
    if positive_indices:
        selected_positive = random.sample(positive_indices, min(num_positive, len(positive_indices)))
        for idx in selected_positive:
            for aug_idx in range(num_augmentations):
                sample = dataset.get_sample_for_visualization(idx)
                img = create_comparison_image(
                    sample["original_image"],
                    sample["augmented_image"],
                    sample["original_coords"],
                    sample["augmented_coords"],
                    sample["has_doc"],
                    sample["is_outlier"],
                    sample["image_name"],
                    img_size,
                )
                filename = f"positive_{Path(sample['image_name']).stem}_aug{aug_idx}.png"
                img.save(output_path / filename)
                visualizations.append(("positive", filename))

    # Negative samples
    print(f"Generating negative sample visualizations...")
    if negative_indices:
        selected_negative = random.sample(negative_indices, min(num_negative, len(negative_indices)))
        for idx in selected_negative:
            for aug_idx in range(num_augmentations):
                sample = dataset.get_sample_for_visualization(idx)
                img = create_comparison_image(
                    sample["original_image"],
                    sample["augmented_image"],
                    sample["original_coords"],
                    sample["augmented_coords"],
                    sample["has_doc"],
                    sample["is_outlier"],
                    sample["image_name"],
                    img_size,
                )
                filename = f"negative_{Path(sample['image_name']).stem}_aug{aug_idx}.png"
                img.save(output_path / filename)
                visualizations.append(("negative", filename))

    # Outlier samples
    print(f"Generating outlier sample visualizations...")
    if outlier_indices:
        selected_outliers = random.sample(outlier_indices, min(num_outliers, len(outlier_indices)))
        for idx in selected_outliers:
            for aug_idx in range(num_augmentations):
                sample = dataset.get_sample_for_visualization(idx)
                img = create_comparison_image(
                    sample["original_image"],
                    sample["augmented_image"],
                    sample["original_coords"],
                    sample["augmented_coords"],
                    sample["has_doc"],
                    sample["is_outlier"],
                    sample["image_name"],
                    img_size,
                )
                filename = f"outlier_{Path(sample['image_name']).stem}_aug{aug_idx}.png"
                img.save(output_path / filename)
                visualizations.append(("outlier", filename))

    # Summary
    print(f"\n{'='*60}")
    print(f"Visualization Summary")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")

    type_counts = {"positive": 0, "negative": 0, "outlier": 0}
    for vtype, _ in visualizations:
        type_counts[vtype] += 1

    print(f"\nGenerated visualizations:")
    print(f"  Positive: {type_counts['positive']}")
    print(f"  Negative: {type_counts['negative']}")
    print(f"  Outlier:  {type_counts['outlier']}")
    print(f"  Total:    {sum(type_counts.values())}")

    # Create index HTML for easy viewing
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>DocCornerNetV3 Augmentation Visualization</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        h2 { color: #666; margin-top: 30px; }
        .gallery { display: flex; flex-wrap: wrap; gap: 10px; }
        .gallery img { max-width: 500px; border: 1px solid #ddd; border-radius: 4px; }
        .legend { margin: 20px 0; padding: 10px; background: #fff; border-radius: 4px; }
        .legend span { margin-right: 20px; }
        .positive { color: #008000; }
        .negative { color: #808080; }
        .outlier { color: #ff8000; }
    </style>
</head>
<body>
    <h1>DocCornerNetV3 Augmentation Visualization</h1>
    <div class="legend">
        <strong>Legend:</strong>
        <span class="positive">Green = Original corners</span>
        <span style="color: red;">Red = Augmented corners</span>
    </div>
"""

    # Positive section
    html_content += "\n    <h2 class='positive'>Positive Samples (with document)</h2>\n    <div class='gallery'>\n"
    for vtype, filename in visualizations:
        if vtype == "positive":
            html_content += f"        <img src='{filename}' alt='{filename}'>\n"
    html_content += "    </div>\n"

    # Negative section
    html_content += "\n    <h2 class='negative'>Negative Samples (no document)</h2>\n    <div class='gallery'>\n"
    for vtype, filename in visualizations:
        if vtype == "negative":
            html_content += f"        <img src='{filename}' alt='{filename}'>\n"
    html_content += "    </div>\n"

    # Outlier section
    html_content += "\n    <h2 class='outlier'>Outlier Samples (harder cases)</h2>\n    <div class='gallery'>\n"
    for vtype, filename in visualizations:
        if vtype == "outlier":
            html_content += f"        <img src='{filename}' alt='{filename}'>\n"
    html_content += "    </div>\n"

    html_content += """
</body>
</html>
"""

    with open(output_path / "index.html", "w") as f:
        f.write(html_content)

    print(f"\nOpen {output_path / 'index.html'} in a browser to view all visualizations.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DocCornerNetV3 augmentation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--data_root", type=str,
                        default="../../datasets/official/doc-scanner-dataset-labeled",
                        help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, default="./visualizations",
                        help="Output directory")
    parser.add_argument("--num_positive", type=int, default=5,
                        help="Number of positive samples")
    parser.add_argument("--num_negative", type=int, default=5,
                        help="Number of negative samples")
    parser.add_argument("--num_outliers", type=int, default=5,
                        help="Number of outlier samples")
    parser.add_argument("--num_augmentations", type=int, default=3,
                        help="Number of augmentation variants per sample")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Image size")
    parser.add_argument("--outlier_list", type=str, default=None,
                        help="Path to outlier list file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Run visualization
    visualize_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        num_positive=args.num_positive,
        num_negative=args.num_negative,
        num_outliers=args.num_outliers,
        num_augmentations=args.num_augmentations,
        img_size=args.img_size,
        outlier_list=args.outlier_list,
    )


if __name__ == "__main__":
    main()
