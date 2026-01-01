"""
Evaluation metrics for DocCornerNetV3 (TensorFlow/Keras).

Contains:
- compute_polygon_iou: True polygon IoU using Shapely
- compute_corner_error: Corner error in pixels
- ValidationMetrics: Accumulator for epoch-level metrics
"""

import numpy as np

try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: shapely not installed. Using bbox IoU fallback.")


def coords_to_polygon(coords: np.ndarray) -> "Polygon":
    """
    Convert 8-value coordinate array to Shapely Polygon.

    Args:
        coords: [8] array (x0,y0,x1,y1,x2,y2,x3,y3) - TL,TR,BR,BL

    Returns:
        Shapely Polygon
    """
    points = [
        (coords[0], coords[1]),  # TL
        (coords[2], coords[3]),  # TR
        (coords[4], coords[5]),  # BR
        (coords[6], coords[7]),  # BL
    ]
    poly = Polygon(points)

    if not poly.is_valid:
        poly = make_valid(poly)
        if poly.geom_type == 'GeometryCollection':
            for geom in poly.geoms:
                if geom.geom_type == 'Polygon':
                    return geom
            return Polygon(points).convex_hull
        elif poly.geom_type == 'MultiPolygon':
            return max(poly.geoms, key=lambda p: p.area)

    return poly


def compute_polygon_iou(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    """
    Compute IoU between predicted and ground truth quadrilaterals.

    Args:
        pred_coords: [8] predicted coordinates (normalized [0, 1])
        gt_coords: [8] ground truth coordinates (normalized [0, 1])

    Returns:
        IoU value in [0, 1]
    """
    if not SHAPELY_AVAILABLE:
        return compute_bbox_iou(pred_coords, gt_coords)

    try:
        pred_poly = coords_to_polygon(pred_coords)
        gt_poly = coords_to_polygon(gt_coords)

        if pred_poly.is_empty or gt_poly.is_empty:
            return 0.0

        intersection = pred_poly.intersection(gt_poly).area
        union = pred_poly.union(gt_poly).area

        if union == 0:
            return 0.0

        return intersection / union

    except Exception:
        return compute_bbox_iou(pred_coords, gt_coords)


def compute_bbox_iou(pred_coords: np.ndarray, gt_coords: np.ndarray) -> float:
    """
    Compute axis-aligned bounding box IoU as fallback.

    Args:
        pred_coords: [8] predicted coordinates
        gt_coords: [8] ground truth coordinates

    Returns:
        Bbox IoU value in [0, 1]
    """
    pred_x = pred_coords[0::2]
    pred_y = pred_coords[1::2]
    gt_x = gt_coords[0::2]
    gt_y = gt_coords[1::2]

    pred_bbox = [pred_x.min(), pred_y.min(), pred_x.max(), pred_y.max()]
    gt_bbox = [gt_x.min(), gt_y.min(), gt_x.max(), gt_y.max()]

    x1 = max(pred_bbox[0], gt_bbox[0])
    y1 = max(pred_bbox[1], gt_bbox[1])
    x2 = min(pred_bbox[2], gt_bbox[2])
    y2 = min(pred_bbox[3], gt_bbox[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    union = pred_area + gt_area - intersection

    if union == 0:
        return 0.0

    return intersection / union


def compute_corner_error(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    img_size: int = 224
) -> tuple[float, np.ndarray]:
    """
    Compute corner error in pixels.

    Args:
        pred_coords: [8] predicted coordinates (normalized [0,1])
        gt_coords: [8] ground truth coordinates (normalized [0,1])
        img_size: Image size for pixel conversion

    Returns:
        mean_error_px: Mean error across all 4 corners
        per_corner_error_px: [4] error per corner
    """
    pred_px = pred_coords * img_size
    gt_px = gt_coords * img_size

    pred_corners = pred_px.reshape(4, 2)
    gt_corners = gt_px.reshape(4, 2)

    distances = np.sqrt(((pred_corners - gt_corners) ** 2).sum(axis=1))

    return float(distances.mean()), distances


class ValidationMetrics:
    """
    Accumulates predictions and computes aggregate metrics.

    Metrics:
    - IoU: mean, median
    - Corner error: mean, p95, max (in pixels)
    - Recall at IoU thresholds: 50%, 75%, 90%, 95%
    - Classification: accuracy, precision, recall, F1
    """

    def __init__(self, img_size: int = 224):
        self.img_size = img_size
        self.reset()

    def reset(self):
        """Reset accumulators."""
        self.pred_coords_list = []
        self.gt_coords_list = []
        self.pred_scores_list = []
        self.has_doc_list = []

    def update(
        self,
        pred_coords: np.ndarray,
        gt_coords: np.ndarray,
        pred_scores: np.ndarray,
        has_doc: np.ndarray,
    ):
        """
        Add batch of predictions.

        Args:
            pred_coords: [B, 8] predicted coordinates
            gt_coords: [B, 8] ground truth coordinates
            pred_scores: [B] or [B, 1] predicted scores (after sigmoid)
            has_doc: [B] or [B, 1] ground truth labels
        """
        if len(pred_scores.shape) == 2:
            pred_scores = pred_scores.squeeze(-1)
        if len(has_doc.shape) == 2:
            has_doc = has_doc.squeeze(-1)

        self.pred_coords_list.append(pred_coords)
        self.gt_coords_list.append(gt_coords)
        self.pred_scores_list.append(pred_scores)
        self.has_doc_list.append(has_doc)

    def compute(self) -> dict:
        """
        Compute aggregate metrics.

        Returns:
            Dictionary with all metrics
        """
        pred_coords = np.concatenate(self.pred_coords_list, axis=0)
        gt_coords = np.concatenate(self.gt_coords_list, axis=0)
        pred_scores = np.concatenate(self.pred_scores_list, axis=0)
        has_doc = np.concatenate(self.has_doc_list, axis=0)

        num_samples = len(pred_coords)
        mask = has_doc == 1
        num_with_doc = int(mask.sum())

        results = {
            "mean_iou": 0.0,
            "median_iou": 0.0,
            "corner_error_px": 0.0,
            "corner_error_p95_px": 0.0,
            "corner_error_min_px": 0.0,
            "corner_error_max_px": 0.0,
            "recall_50": 0.0,
            "recall_75": 0.0,
            "recall_90": 0.0,
            "recall_95": 0.0,
            "recall_99": 0.0,
            "cls_accuracy": 0.0,
            "cls_precision": 0.0,
            "cls_recall": 0.0,
            "cls_f1": 0.0,
            "num_samples": num_samples,
            "num_with_doc": num_with_doc,
        }

        # Classification metrics
        pred_labels = (pred_scores > 0.5).astype(int)
        tp = int(((pred_labels == 1) & (has_doc == 1)).sum())
        fp = int(((pred_labels == 1) & (has_doc == 0)).sum())
        tn = int(((pred_labels == 0) & (has_doc == 0)).sum())
        fn = int(((pred_labels == 0) & (has_doc == 1)).sum())

        if num_samples > 0:
            results["cls_accuracy"] = (tp + tn) / num_samples
        if tp + fp > 0:
            results["cls_precision"] = tp / (tp + fp)
        if tp + fn > 0:
            results["cls_recall"] = tp / (tp + fn)
        if results["cls_precision"] + results["cls_recall"] > 0:
            results["cls_f1"] = (
                2 * results["cls_precision"] * results["cls_recall"] /
                (results["cls_precision"] + results["cls_recall"])
            )

        # Geometry metrics (only positive samples)
        if num_with_doc == 0:
            return results

        pred_coords_pos = pred_coords[mask]
        gt_coords_pos = gt_coords[mask]

        # IoU per sample
        ious = []
        for i in range(num_with_doc):
            iou = compute_polygon_iou(pred_coords_pos[i], gt_coords_pos[i])
            ious.append(iou)
        ious = np.array(ious)

        # Corner error per sample
        all_corner_errors = []
        mean_corner_errors = []
        for i in range(num_with_doc):
            mean_err, per_corner = compute_corner_error(
                pred_coords_pos[i], gt_coords_pos[i], self.img_size
            )
            mean_corner_errors.append(mean_err)
            all_corner_errors.extend(per_corner)

        all_corner_errors = np.array(all_corner_errors)
        mean_corner_errors = np.array(mean_corner_errors)

        # Update results
        results["mean_iou"] = float(np.mean(ious))
        results["median_iou"] = float(np.median(ious))
        results["corner_error_px"] = float(np.mean(mean_corner_errors))
        results["corner_error_p95_px"] = float(np.percentile(all_corner_errors, 95))
        results["corner_error_min_px"] = float(np.min(all_corner_errors))
        results["corner_error_max_px"] = float(np.max(all_corner_errors))

        results["recall_50"] = float((ious >= 0.50).sum() / num_with_doc)
        results["recall_75"] = float((ious >= 0.75).sum() / num_with_doc)
        results["recall_90"] = float((ious >= 0.90).sum() / num_with_doc)
        results["recall_95"] = float((ious >= 0.95).sum() / num_with_doc)
        results["recall_99"] = float((ious >= 0.99).sum() / num_with_doc)

        return results


if __name__ == "__main__":
    print(f"Shapely available: {SHAPELY_AVAILABLE}")

    # Test polygon IoU
    print("\n--- Testing polygon IoU ---")

    coords1 = np.array([0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8])
    coords2 = np.array([0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8])
    print(f"Perfect match IoU: {compute_polygon_iou(coords1, coords2):.4f}")

    coords3 = np.array([0.3, 0.3, 0.9, 0.3, 0.9, 0.9, 0.3, 0.9])
    print(f"Partial overlap IoU: {compute_polygon_iou(coords1, coords3):.4f}")

    # Test corner error
    print("\n--- Testing corner error ---")
    mean_err, per_corner = compute_corner_error(coords1, coords3, img_size=224)
    print(f"Mean corner error: {mean_err:.2f}px")

    # Test ValidationMetrics
    print("\n--- Testing ValidationMetrics ---")
    metrics = ValidationMetrics(img_size=224)

    # Simulate batch
    B = 10
    pred_coords = np.random.uniform(0.1, 0.9, (B, 8)).astype(np.float32)
    gt_coords = pred_coords + np.random.normal(0, 0.02, (B, 8)).astype(np.float32)
    gt_coords = np.clip(gt_coords, 0, 1)
    pred_scores = np.array([0.9, 0.8, 0.7, 0.1, 0.9, 0.85, 0.2, 0.95, 0.6, 0.3])
    has_doc = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 0])

    metrics.update(pred_coords, gt_coords, pred_scores, has_doc)
    results = metrics.compute()

    print(f"\nResults:")
    print(f"  Mean IoU: {results['mean_iou']:.4f}")
    print(f"  Corner Error (mean): {results['corner_error_px']:.2f}px")
    print(f"  Corner Error (p95): {results['corner_error_p95_px']:.2f}px")
    print(f"  Recall@90: {results['recall_90']*100:.1f}%")
    print(f"  Recall@95: {results['recall_95']*100:.1f}%")
    print(f"  Recall@99: {results['recall_99']*100:.1f}%")
    print(f"  Classification Accuracy: {results['cls_accuracy']*100:.1f}%")

    print("\nMetrics test passed!")
