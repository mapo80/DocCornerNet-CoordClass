"""
Evaluate TFLite models on validation dataset.

Metrics:
- IoU (Intersection over Union)
- Pixel Error (L2 distance in pixels)
- Recall@95 (% samples with IoU >= 0.95)
- Inference time
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TFLite models on validation dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--tflite_models", type=str, nargs="+", required=True,
                        help="Paths to TFLite models to evaluate")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset root (with images/ and labels.json)")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Max samples to evaluate (None = all)")
    parser.add_argument("--benchmark_runs", type=int, default=50,
                        help="Number of runs for latency benchmark")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")

    return parser.parse_args()


def load_labels(data_root: Path, split: str):
    """Load labels for the specified split.

    Supports two formats:
    1. labels.json with {filename: {corners: [[x,y],...], split: "val"}}
    2. {split}.txt with filenames, and labels/ folder with YOLO-style txt files
    """
    labels_json = data_root / "labels.json"
    split_txt = data_root / f"{split}.txt"

    samples = []

    if labels_json.exists():
        # JSON format
        with open(labels_json) as f:
            all_labels = json.load(f)

        for filename, data in all_labels.items():
            if data.get("split") == split:
                corners = data.get("corners")
                if corners:
                    coords = np.array(corners, dtype=np.float32).flatten()
                    samples.append({
                        "filename": filename,
                        "coords": coords,
                        "has_document": True
                    })

    elif split_txt.exists():
        # TXT format (YOLO-style)
        labels_dir = data_root / "labels"

        with open(split_txt) as f:
            filenames = [line.strip() for line in f if line.strip()]

        for filename in filenames:
            # Get corresponding label file
            label_file = labels_dir / filename.replace('.jpg', '.txt').replace('.png', '.txt')

            if not label_file.exists():
                continue

            with open(label_file) as f:
                line = f.readline().strip()

            if not line:
                continue

            parts = line.split()
            if len(parts) < 9:  # class + 8 coords
                continue

            # Format: class x0 y0 x1 y1 x2 y2 x3 y3
            coords = np.array([float(x) for x in parts[1:9]], dtype=np.float32)

            samples.append({
                "filename": filename,
                "coords": coords,
                "has_document": True
            })

    else:
        raise FileNotFoundError(f"No labels.json or {split}.txt found in {data_root}")

    return samples


def preprocess_image(image_path: Path, img_size: int) -> np.ndarray:
    """Load and preprocess image for inference."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    return img[np.newaxis, ...]  # Add batch dimension


def compute_iou(pred_coords: np.ndarray, gt_coords: np.ndarray, img_size: int = 224) -> float:
    """
    Compute IoU between predicted and ground truth quadrilaterals.

    Args:
        pred_coords: [8] - predicted normalized coordinates (x0,y0,x1,y1,x2,y2,x3,y3)
        gt_coords: [8] - ground truth normalized coordinates
        img_size: image size for pixel conversion

    Returns:
        IoU score (0-1)
    """
    try:
        from shapely.geometry import Polygon

        # Convert to pixel coordinates
        pred_pts = pred_coords.reshape(4, 2) * img_size
        gt_pts = gt_coords.reshape(4, 2) * img_size

        # Create polygons
        pred_poly = Polygon(pred_pts)
        gt_poly = Polygon(gt_pts)

        if not pred_poly.is_valid or not gt_poly.is_valid:
            return 0.0

        # Compute IoU
        intersection = pred_poly.intersection(gt_poly).area
        union = pred_poly.union(gt_poly).area

        if union == 0:
            return 0.0

        return intersection / union

    except Exception as e:
        # Fallback: simple bounding box IoU
        return compute_bbox_iou(pred_coords, gt_coords)


def compute_bbox_iou(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    """Fallback: compute bounding box IoU."""
    pred_pts = pred_coords.reshape(4, 2)
    gt_pts = gt_coords.reshape(4, 2)

    # Get bounding boxes
    pred_x1, pred_y1 = pred_pts.min(axis=0)
    pred_x2, pred_y2 = pred_pts.max(axis=0)
    gt_x1, gt_y1 = gt_pts.min(axis=0)
    gt_x2, gt_y2 = gt_pts.max(axis=0)

    # Intersection
    inter_x1 = max(pred_x1, gt_x1)
    inter_y1 = max(pred_y1, gt_y1)
    inter_x2 = min(pred_x2, gt_x2)
    inter_y2 = min(pred_y2, gt_y2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    union_area = pred_area + gt_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def compute_pixel_error(pred_coords: np.ndarray, gt_coords: np.ndarray, img_size: int = 224) -> float:
    """Compute mean L2 pixel error across all corners."""
    pred_px = pred_coords * img_size
    gt_px = gt_coords * img_size

    # Reshape to [4, 2] and compute per-corner distances
    pred_pts = pred_px.reshape(4, 2)
    gt_pts = gt_px.reshape(4, 2)

    distances = np.linalg.norm(pred_pts - gt_pts, axis=1)
    return float(np.mean(distances))


class TFLiteModel:
    """Wrapper for TFLite model inference."""

    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Identify output indices
        self.coords_idx = None
        self.score_idx = None

        for i, od in enumerate(self.output_details):
            shape = od['shape']
            if len(shape) == 2:
                if shape[1] == 8:
                    self.coords_idx = od['index']
                elif shape[1] == 1:
                    self.score_idx = od['index']

        if self.coords_idx is None:
            raise ValueError("Could not find coords output (shape [N, 8])")

    def predict(self, image: np.ndarray) -> dict:
        """Run inference on a single image."""
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()

        coords = self.interpreter.get_tensor(self.coords_idx)
        score = self.interpreter.get_tensor(self.score_idx) if self.score_idx else None

        return {
            'coords': coords[0],  # [8]
            'score': float(score[0, 0]) if score is not None else 0.0
        }

    def benchmark(self, img_size: int, num_runs: int = 100) -> dict:
        """Benchmark inference latency."""
        dummy = np.random.randn(1, img_size, img_size, 3).astype(np.float32)

        # Warmup
        for _ in range(10):
            self.predict(dummy)

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.predict(dummy)
            times.append(time.perf_counter() - start)

        times_ms = np.array(times) * 1000

        return {
            'mean_ms': float(np.mean(times_ms)),
            'std_ms': float(np.std(times_ms)),
            'p50_ms': float(np.percentile(times_ms, 50)),
            'p95_ms': float(np.percentile(times_ms, 95)),
            'min_ms': float(np.min(times_ms)),
            'max_ms': float(np.max(times_ms)),
        }


def evaluate_model(model: TFLiteModel, samples: list, data_root: Path,
                   img_size: int, num_samples: int = None) -> dict:
    """Evaluate model on samples."""
    if num_samples:
        samples = samples[:num_samples]

    ious = []
    pixel_errors = []

    image_dir = data_root / "images"

    print(f"  Evaluating {len(samples)} samples...")

    for i, sample in enumerate(samples):
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(samples)}")

        # Load and preprocess image
        img_path = image_dir / sample['filename']
        if not img_path.exists():
            continue

        image = preprocess_image(img_path, img_size)

        # Run inference
        output = model.predict(image)
        pred_coords = output['coords']
        gt_coords = sample['coords']

        # Compute metrics
        iou = compute_iou(pred_coords, gt_coords, img_size)
        px_err = compute_pixel_error(pred_coords, gt_coords, img_size)

        ious.append(iou)
        pixel_errors.append(px_err)

    ious = np.array(ious)
    pixel_errors = np.array(pixel_errors)

    # Compute aggregate metrics
    results = {
        'num_samples': len(ious),
        'iou_mean': float(np.mean(ious)),
        'iou_std': float(np.std(ious)),
        'iou_median': float(np.median(ious)),
        'iou_min': float(np.min(ious)),
        'iou_max': float(np.max(ious)),
        'pixel_error_mean': float(np.mean(pixel_errors)),
        'pixel_error_std': float(np.std(pixel_errors)),
        'pixel_error_median': float(np.median(pixel_errors)),
        'recall_90': float(np.mean(ious >= 0.90) * 100),
        'recall_95': float(np.mean(ious >= 0.95) * 100),
        'recall_99': float(np.mean(ious >= 0.99) * 100),
    }

    return results


def main():
    args = parse_args()

    data_root = Path(args.data_root)

    print("=" * 70)
    print("TFLite Model Evaluation")
    print("=" * 70)

    # Load validation samples
    print(f"\nLoading {args.split} samples from {data_root}...")
    samples = load_labels(data_root, args.split)
    print(f"  Found {len(samples)} samples")

    if len(samples) == 0:
        print("ERROR: No samples found!")
        return

    all_results = {}

    for model_path in args.tflite_models:
        model_path = Path(model_path)
        model_name = model_path.stem

        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"  Path: {model_path}")
        print(f"  Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
        print("=" * 70)

        # Load model
        model = TFLiteModel(str(model_path))

        # Benchmark latency
        print("\nBenchmarking latency...")
        latency = model.benchmark(args.img_size, args.benchmark_runs)
        print(f"  Mean: {latency['mean_ms']:.2f} ms")
        print(f"  P50:  {latency['p50_ms']:.2f} ms")
        print(f"  P95:  {latency['p95_ms']:.2f} ms")

        # Evaluate accuracy
        print("\nEvaluating accuracy...")
        metrics = evaluate_model(model, samples, data_root, args.img_size, args.num_samples)

        print(f"\nResults:")
        print(f"  IoU:        {metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f}")
        print(f"  Pixel Err:  {metrics['pixel_error_mean']:.2f} ± {metrics['pixel_error_std']:.2f} px")
        print(f"  R@90:       {metrics['recall_90']:.1f}%")
        print(f"  R@95:       {metrics['recall_95']:.1f}%")
        print(f"  R@99:       {metrics['recall_99']:.1f}%")

        all_results[model_name] = {
            'path': str(model_path),
            'size_mb': model_path.stat().st_size / (1024*1024),
            'latency': latency,
            'metrics': metrics,
        }

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Size':>8} {'Latency':>10} {'IoU':>8} {'Err(px)':>8} {'R@95':>8}")
    print("-" * 70)

    for name, data in all_results.items():
        print(f"{name:<25} {data['size_mb']:>7.2f}M {data['latency']['mean_ms']:>9.2f}ms "
              f"{data['metrics']['iou_mean']:>7.4f} {data['metrics']['pixel_error_mean']:>7.2f} "
              f"{data['metrics']['recall_95']:>7.1f}%")

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
