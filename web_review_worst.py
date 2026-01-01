"""
Worst Cases Review Web Application for DocCornerNet.

Simplified Gradio interface for reviewing worst cases from evaluate_worst_cases.py output.
Allows modifying GT bbox or discarding images from dataset.

Features:
- Load worst cases from worst_cases.txt
- View GT bbox (green) on original image
- Edit GT corners via click or sliders
- Discard images from train/val splits
- Export modified labels and cleaned split files

Usage:
    python web_review_worst.py \
        --worst_cases ./evaluation_results/worst_cases.txt \
        --dataset ./datasets/official/doc-scanner-dataset-rev-new \
        --images_tar ./evaluation_results/worst_images.tar.gz
"""

import argparse
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import gradio as gr
except ImportError:
    print("Error: gradio not installed. Run: pip install gradio")
    sys.exit(1)

# Constants
CORNER_LABELS = ["TL", "TR", "BR", "BL"]
CORNER_CLICK_RADIUS = 20
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp", ".WEBP"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class WorstCaseReviewApp:
    """Web application for reviewing worst case predictions."""

    def __init__(
        self,
        worst_cases_path: str,
        dataset_path: str,
        images_tar_path: Optional[str] = None,
        display_size: int = 512,
        inference_model=None,
        inference_img_size: Optional[int] = None,
        input_norm: str = "imagenet",
    ):
        self.worst_cases_path = Path(worst_cases_path)
        self.dataset_path = Path(dataset_path)
        self.images_tar_path = Path(images_tar_path) if images_tar_path else None
        self.display_size = display_size

        # State
        self.entries: List[Dict] = []
        self.gt_labels: Dict[str, np.ndarray] = {}
        self.modified: Dict[str, np.ndarray] = {}
        self.discarded: set = set()

        self.current_index = 0
        self.filtered_list: List[str] = []
        self.current_filter = "all"
        self.current_sort = "error_desc"
        self.status_message = ""

        # For click editing
        self.selected_corner: Optional[int] = None

        # Pre-rendered images cache (from tar)
        self.preview_images: Dict[str, Path] = {}
        self.preview_dir: Optional[Path] = None
        self.inference_model = inference_model
        self.inference_img_size = inference_img_size
        self.input_norm = (input_norm or "imagenet").lower().strip()
        self.pred_cache: Dict[str, np.ndarray] = {}

        # Load data
        self._load_worst_cases()
        self._load_gt_labels()
        self._extract_preview_images()
        self._load_discarded()
        self._apply_filter_and_sort()

    def _resolve_image_path(self, name: str) -> Optional[Path]:
        stem = Path(name).stem
        images_dir = self.dataset_path / "images"
        for ext in IMAGE_EXTS:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _normalize_for_model(self, image: np.ndarray) -> np.ndarray:
        x = image.astype(np.float32)
        norm = self.input_norm
        if norm in {"raw255", "0_255", "0255"}:
            return x
        x = x / 255.0
        if norm in {"zero_one", "0_1", "01"}:
            return x
        return (x - IMAGENET_MEAN) / IMAGENET_STD

    def _get_pred_coords(self, name: str) -> Optional[np.ndarray]:
        if self.inference_model is None:
            return None
        if name in self.pred_cache:
            return self.pred_cache[name]

        img_path = self._resolve_image_path(name)
        if img_path is None:
            return None

        img = cv2.imread(str(img_path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        size = self.inference_img_size or img.shape[0]
        img = cv2.resize(img, (size, size))
        x = self._normalize_for_model(img)
        x = x[None, ...]

        preds = self.inference_model(x, training=False)
        if isinstance(preds, dict):
            coords = preds.get("coords")
        elif isinstance(preds, (list, tuple)):
            coords = preds[0]
        else:
            coords = preds

        if coords is None:
            return None

        coords = coords.numpy()[0]
        coords = np.clip(coords, 0.0, 1.0)
        self.pred_cache[name] = coords
        return coords

    def _load_worst_cases(self):
        """Load worst cases from txt file."""
        if not self.worst_cases_path.exists():
            raise FileNotFoundError(f"Worst cases file not found: {self.worst_cases_path}")

        with open(self.worst_cases_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('=') or line.startswith('Split') or line.startswith('SUMMARY'):
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        self.entries.append({
                            'split': parts[0],
                            'name': parts[1],
                            'iou': float(parts[2]),
                            'err_mean': float(parts[3]),
                            'err_max': float(parts[4]),
                            'score': float(parts[5]) if len(parts) > 5 else 1.0,
                        })
                    except:
                        continue

        print(f"Loaded {len(self.entries)} worst cases from {self.worst_cases_path}")

    def _load_gt_labels(self):
        """Load GT labels from dataset."""
        labels_dir = self.dataset_path / "labels"
        if not labels_dir.exists():
            print(f"Warning: labels directory not found: {labels_dir}")
            return

        for entry in self.entries:
            name = entry['name']
            stem = Path(name).stem
            label_path = labels_dir / f"{stem}.txt"

            if label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        line = f.readline().strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 9 and parts[0] == '0':
                                coords = np.array([float(parts[i]) for i in range(1, 9)], dtype=np.float32)
                                self.gt_labels[name] = coords
                except Exception as e:
                    print(f"Warning: failed to load label for {name}: {e}")

        print(f"Loaded {len(self.gt_labels)} GT labels")

    def _extract_preview_images(self):
        """Extract preview images from tar if provided."""
        if self.images_tar_path is None or not self.images_tar_path.exists():
            print("No images tar provided, will load from dataset")
            return

        # Extract to temp dir
        self.preview_dir = self.worst_cases_path.parent / "worst_images_extracted"

        if not self.preview_dir.exists():
            print(f"Extracting preview images to {self.preview_dir}...")
            self.preview_dir.mkdir(parents=True, exist_ok=True)

            with tarfile.open(self.images_tar_path, 'r:gz') as tar:
                tar.extractall(self.preview_dir.parent)

            # Handle nested directory
            nested = self.preview_dir.parent / "worst_images"
            if nested.exists() and nested.is_dir():
                for f in nested.iterdir():
                    shutil.move(str(f), str(self.preview_dir / f.name))
                nested.rmdir()

        # Index preview images
        for img_path in self.preview_dir.glob("*.jpg"):
            # Parse filename: 0001_err15.2px_iou0.850_imagename.jpg
            fname = img_path.stem
            parts = fname.split('_', 3)
            if len(parts) >= 4:
                image_name = parts[3]
                # Find matching entry
                for entry in self.entries:
                    if Path(entry['name']).stem == image_name:
                        self.preview_images[entry['name']] = img_path
                        break

        print(f"Indexed {len(self.preview_images)} preview images")

    def _load_discarded(self):
        """Load discarded list if exists."""
        discarded_path = self.dataset_path / "discarded_worst.txt"
        if discarded_path.exists():
            with open(discarded_path, 'r') as f:
                for line in f:
                    name = line.strip()
                    if name:
                        self.discarded.add(name)
            print(f"Loaded {len(self.discarded)} previously discarded images")

    def _apply_filter_and_sort(self):
        """Apply current filter and sort."""
        # Filter
        if self.current_filter == "all":
            filtered = [e['name'] for e in self.entries if e['name'] not in self.discarded]
        elif self.current_filter == "train":
            filtered = [e['name'] for e in self.entries
                       if e['split'] == 'train' and e['name'] not in self.discarded]
        elif self.current_filter == "val":
            filtered = [e['name'] for e in self.entries
                       if e['split'] == 'val' and e['name'] not in self.discarded]
        elif self.current_filter == "modified":
            filtered = [e['name'] for e in self.entries
                       if e['name'] in self.modified and e['name'] not in self.discarded]
        elif self.current_filter == "discarded":
            filtered = [e['name'] for e in self.entries if e['name'] in self.discarded]
        elif self.current_filter == "high_error":
            filtered = [e['name'] for e in self.entries
                       if e['err_mean'] > 20 and e['name'] not in self.discarded]
        else:
            filtered = [e['name'] for e in self.entries if e['name'] not in self.discarded]

        # Get entry map for sorting
        entry_map = {e['name']: e for e in self.entries}

        # Sort
        if self.current_sort == "error_desc":
            filtered.sort(key=lambda n: -entry_map[n]['err_mean'])
        elif self.current_sort == "error_asc":
            filtered.sort(key=lambda n: entry_map[n]['err_mean'])
        elif self.current_sort == "iou_asc":
            filtered.sort(key=lambda n: entry_map[n]['iou'])
        elif self.current_sort == "iou_desc":
            filtered.sort(key=lambda n: -entry_map[n]['iou'])
        elif self.current_sort == "name":
            filtered.sort()

        self.filtered_list = filtered
        self.current_index = min(self.current_index, max(0, len(filtered) - 1))

    def get_current_entry(self) -> Optional[Dict]:
        """Get current entry data."""
        if not self.filtered_list:
            return None
        name = self.filtered_list[self.current_index]
        for e in self.entries:
            if e['name'] == name:
                return e
        return None

    def get_current_corners(self) -> np.ndarray:
        """Get current GT corners (modified if available)."""
        if not self.filtered_list:
            return np.zeros(8)
        name = self.filtered_list[self.current_index]
        if name in self.modified:
            return self.modified[name]
        return self.gt_labels.get(name, np.zeros(8))

    def get_dashboard_stats(self) -> str:
        """Get aggregate statistics."""
        total = len(self.entries)
        train_count = sum(1 for e in self.entries if e['split'] == 'train' and e['name'] not in self.discarded)
        val_count = sum(1 for e in self.entries if e['split'] == 'val' and e['name'] not in self.discarded)
        modified_count = len(self.modified)
        discarded_count = len(self.discarded)

        active = [e for e in self.entries if e['name'] not in self.discarded]
        if active:
            mean_err = np.mean([e['err_mean'] for e in active])
            mean_iou = np.mean([e['iou'] for e in active])
        else:
            mean_err = 0
            mean_iou = 0

        return f"""### Worst Cases Review Dashboard

| Metric | Value |
|--------|-------|
| **Total Worst Cases** | {total} |
| **Train Split** | {train_count} |
| **Val Split** | {val_count} |
| **Mean Error (active)** | {mean_err:.1f}px |
| **Mean IoU (active)** | {mean_iou:.4f} |
| **Modified** | {modified_count} |
| **Discarded** | {discarded_count} |
"""

    def get_current_image_info(self) -> str:
        """Get info for current image."""
        if not self.filtered_list:
            return "No images match filter."

        name = self.filtered_list[self.current_index]
        entry = self.get_current_entry()
        if entry is None:
            return "Entry not found."

        tags = []
        if name in self.discarded:
            tags.append("DISCARDED")
        if name in self.modified:
            tags.append("MODIFIED")
        status = " ".join(f"**[{t}]**" for t in tags)

        return f"""### {name} {status}

| Metric | Value |
|--------|-------|
| **Split** | {entry['split'].upper()} |
| **IoU** | {entry['iou']:.4f} |
| **Error (mean)** | {entry['err_mean']:.1f}px |
| **Error (max)** | {entry['err_max']:.1f}px |
| **Score** | {entry['score']:.4f} |

Position: {self.current_index + 1} / {len(self.filtered_list)}
"""

    def render_image(self) -> np.ndarray:
        """Render current image with GT overlay."""
        if not self.filtered_list:
            # Return placeholder
            img = np.ones((self.display_size, self.display_size, 3), dtype=np.uint8) * 128
            cv2.putText(img, "No images", (self.display_size//4, self.display_size//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return img

        name = self.filtered_list[self.current_index]

        # Try to load image
        img = None

        # First try preview image (only if no inference overlay needed)
        if name in self.preview_images and self.inference_model is None:
            img_path = self.preview_images[name]
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Preview images already have overlay, just resize
                img = cv2.resize(img, (self.display_size, self.display_size + 70))
                return img

        # Try dataset images
        img_path = self._resolve_image_path(name)
        if img_path is not None:
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
            # Placeholder
            img = np.ones((self.display_size, self.display_size, 3), dtype=np.uint8) * 64
            cv2.putText(img, f"Image not found: {name}", (10, self.display_size//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            return img

        # Resize
        img = cv2.resize(img, (self.display_size, self.display_size))

        # Draw GT (green)
        corners = self.get_current_corners()
        if corners.sum() > 0:
            self._draw_polygon(img, corners, (0, 255, 0), "GT", draw_handles=True, draw_labels=True)
            cv2.putText(img, "GT", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw inference (red)
        pred = self._get_pred_coords(name)
        if pred is not None and pred.sum() > 0:
            self._draw_polygon(img, pred, (255, 0, 0), "PRED", draw_handles=False, draw_labels=False)
            cv2.putText(img, "PRED", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return img

    def _draw_polygon(
        self,
        img: np.ndarray,
        coords: np.ndarray,
        color: Tuple[int, int, int],
        label: str,
        draw_handles: bool = False,
        draw_labels: bool = True,
    ):
        """Draw polygon on image."""
        h, w = img.shape[:2]
        points = []
        for i in range(4):
            x = int(coords[i * 2] * w)
            y = int(coords[i * 2 + 1] * h)
            points.append((x, y))

        # Draw lines
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            cv2.line(img, p1, p2, color, thickness=2)

        # Draw corner circles and labels
        for i, (x, y) in enumerate(points):
            if draw_handles:
                radius = 14 if self.selected_corner == i else 10
                if self.selected_corner == i:
                    cv2.circle(img, (x, y), radius + 4, (255, 255, 0), 3)
                cv2.circle(img, (x, y), radius, color, -1)
                cv2.circle(img, (x, y), radius, (255, 255, 255), 2)
                cv2.circle(img, (x, y), 3, (255, 255, 255), -1)
            else:
                cv2.circle(img, (x, y), 5, color, -1)

            if draw_labels:
                label_pos = (x + 12, y - 12)
                cv2.putText(img, CORNER_LABELS[i], label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(img, CORNER_LABELS[i], label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def save_gt_changes(self, corners: np.ndarray):
        """Save modified GT corners."""
        if not self.filtered_list:
            return
        name = self.filtered_list[self.current_index]
        corners = np.clip(np.array(corners, dtype=np.float32), 0.0, 1.0)
        self.modified[name] = corners
        self.status_message = f"Modified GT for {name}"

    def discard_current(self):
        """Discard current image and move to next."""
        if not self.filtered_list:
            return
        name = self.filtered_list[self.current_index]
        if name in self.discarded:
            return
        self.discarded.add(name)

        # Remove from entries list so it won't appear again
        self.entries = [e for e in self.entries if e['name'] != name]

        # Also remove from preview cache
        if name in self.preview_images:
            del self.preview_images[name]

        # Update worst_cases.txt immediately (remove discarded entry)
        self._update_worst_cases_file()

        # Save discarded list to disk
        self._save_discarded_list()

        self.status_message = f"Discarded {name} (removed permanently)"
        self._apply_filter_and_sort()

        # Stay at same index (which now shows next image) or adjust if at end
        if self.current_index >= len(self.filtered_list):
            self.current_index = max(0, len(self.filtered_list) - 1)

    def _update_worst_cases_file(self):
        """Rewrite worst_cases.txt without discarded entries."""
        if not self.worst_cases_path.exists():
            return
        try:
            # Write remaining entries
            with open(self.worst_cases_path, 'w') as f:
                f.write(f"{'Split':<8} {'Filename':<60} {'IoU':>8} {'Err_mean':>10} {'Err_max':>10} {'Score':>8}\n")
                f.write("=" * 110 + "\n")
                for e in self.entries:
                    f.write(f"{e['split']:<8} {e['name']:<60} {e['iou']:>8.4f} {e['err_mean']:>10.2f} {e['err_max']:>10.2f} {e['score']:>8.4f}\n")
        except Exception as ex:
            print(f"Failed to update worst_cases.txt: {ex}")

    def _save_discarded_list(self):
        """Save discarded list to dataset directory."""
        try:
            discarded_path = self.dataset_path / "discarded_worst.txt"
            self.dataset_path.mkdir(parents=True, exist_ok=True)
            with open(discarded_path, 'w') as f:
                for name in sorted(self.discarded):
                    f.write(name + "\n")
        except Exception as ex:
            print(f"Failed to save discarded list: {ex}")

    def undiscard_current(self):
        """Restore discarded image."""
        if not self.filtered_list:
            return
        name = self.filtered_list[self.current_index]
        if name not in self.discarded:
            return
        self.discarded.remove(name)
        self.status_message = f"Restored {name}"
        self._apply_filter_and_sort()

    def undo_modifications(self):
        """Undo modifications for current image."""
        if not self.filtered_list:
            return
        name = self.filtered_list[self.current_index]
        if name in self.modified:
            del self.modified[name]
            self.status_message = f"Undid modifications for {name}"

    def save_all_to_disk(self) -> str:
        """Save all changes to disk."""
        saved_labels = 0

        # Ensure dataset directory exists
        if not self.dataset_path.exists():
            return f"Error: Dataset path does not exist: {self.dataset_path}"

        # Save modified labels
        labels_dir = self.dataset_path / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        for name, coords in self.modified.items():
            stem = Path(name).stem
            label_path = labels_dir / f"{stem}.txt"
            try:
                with open(label_path, 'w') as f:
                    line = "0 " + " ".join(f"{c:.6f}" for c in coords)
                    f.write(line + "\n")
                saved_labels += 1
            except Exception as e:
                print(f"Failed to save label for {name}: {e}")

        # Save discarded list
        discarded_path = self.dataset_path / "discarded_worst.txt"
        with open(discarded_path, 'w') as f:
            for name in sorted(self.discarded):
                f.write(name + "\n")

        self.status_message = f"Saved {saved_labels} modified labels, {len(self.discarded)} discarded"
        return self.status_message

    def export_clean_splits(self) -> str:
        """Export cleaned train.txt and val.txt without discarded images."""
        results = []

        for split in ['train', 'val']:
            split_file = self.dataset_path / f"{split}.txt"
            if not split_file.exists():
                continue

            with open(split_file, 'r') as f:
                lines = f.readlines()

            # Filter out discarded
            clean_lines = []
            removed = 0
            for line in lines:
                name = line.strip()
                if not name:
                    continue
                # Check if discarded
                if name in self.discarded:
                    removed += 1
                    continue
                # Also check with different path formats
                check_variants = [
                    name,
                    name.replace("images/", ""),
                    Path(name).name,
                ]
                is_discarded = any(v in self.discarded for v in check_variants)
                if is_discarded:
                    removed += 1
                    continue
                clean_lines.append(line)

            # Save cleaned split
            clean_path = self.dataset_path / f"{split}_cleaned.txt"
            with open(clean_path, 'w') as f:
                f.writelines(clean_lines)

            results.append(f"{split}: {len(clean_lines)} kept, {removed} removed -> {clean_path.name}")

        self.status_message = "; ".join(results)
        return self.status_message


def create_app(app: WorstCaseReviewApp) -> gr.Blocks:
    """Create Gradio interface."""

    def refresh_all():
        corners = app.get_current_corners()
        img = app.render_image()
        info = app.get_current_image_info()
        stats = app.get_dashboard_stats()
        pos = f"{app.current_index + 1}" if app.filtered_list else "0"
        total = f"/ {len(app.filtered_list)}"

        return (
            img, info, stats,
            float(corners[0]), float(corners[1]),  # TL
            float(corners[2]), float(corners[3]),  # TR
            float(corners[4]), float(corners[5]),  # BR
            float(corners[6]), float(corners[7]),  # BL
            pos, total, app.status_message,
        )

    def update_filter(filter_choice):
        filter_map = {
            "All": "all",
            "Train only": "train",
            "Val only": "val",
            "High Error (>20px)": "high_error",
            "Modified": "modified",
            "Discarded": "discarded",
        }
        app.current_filter = filter_map.get(filter_choice, "all")
        app.current_index = 0
        app.status_message = ""
        app._apply_filter_and_sort()
        return refresh_all()

    def update_sort(sort_choice):
        sort_map = {
            "Error (high to low)": "error_desc",
            "Error (low to high)": "error_asc",
            "IoU (low to high)": "iou_asc",
            "IoU (high to low)": "iou_desc",
            "Filename": "name",
        }
        app.current_sort = sort_map.get(sort_choice, "error_desc")
        app.status_message = ""
        app._apply_filter_and_sort()
        return refresh_all()

    def go_prev():
        if app.filtered_list:
            app.current_index = max(0, app.current_index - 1)
        app.status_message = ""
        return refresh_all()

    def go_next():
        if app.filtered_list:
            app.current_index = min(len(app.filtered_list) - 1, app.current_index + 1)
        app.status_message = ""
        return refresh_all()

    def go_to_index(idx):
        try:
            idx = int(idx) - 1
            if app.filtered_list:
                app.current_index = max(0, min(len(app.filtered_list) - 1, idx))
        except ValueError:
            pass
        app.status_message = ""
        return refresh_all()

    def apply_corners(tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y):
        corners = np.array([tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y], dtype=np.float32)
        app.save_gt_changes(corners)
        return refresh_all()

    def discard_image():
        app.discard_current()
        return refresh_all()

    def undiscard_image():
        app.undiscard_current()
        return refresh_all()

    def undo_changes():
        app.undo_modifications()
        return refresh_all()

    def save_all():
        result = app.save_all_to_disk()
        stats = app.get_dashboard_stats()
        return stats, result

    def export_splits():
        result = app.export_clean_splits()
        return result

    # Build interface
    with gr.Blocks(title="Worst Cases Review", css="""
        .corner-slider { max-width: 120px !important; }
    """) as demo:
        gr.Markdown("# DocCornerNet Worst Cases Review")

        with gr.Row():
            # Left column - controls
            with gr.Column(scale=1):
                gr.Markdown("### Filters & Navigation")

                filter_dropdown = gr.Dropdown(
                    choices=["All", "Train only", "Val only", "High Error (>20px)", "Modified", "Discarded"],
                    value="All",
                    label="Filter",
                )
                sort_dropdown = gr.Dropdown(
                    choices=["Error (high to low)", "Error (low to high)", "IoU (low to high)", "IoU (high to low)", "Filename"],
                    value="Error (high to low)",
                    label="Sort",
                )

                discard_btn = gr.Button("🗑️ Discard", variant="stop")

                with gr.Row():
                    prev_btn = gr.Button("◀ Prev", size="sm")
                    next_btn = gr.Button("Next ▶", size="sm")

                with gr.Row():
                    position_input = gr.Textbox(value="1", label="Position", max_lines=1, scale=1)
                    total_label = gr.Markdown(f"/ {len(app.filtered_list)}")

                gr.Markdown("---")
                gr.Markdown("### Actions")

                undiscard_btn = gr.Button("♻️ Restore", variant="secondary")
                undo_btn = gr.Button("↩️ Undo Changes", variant="secondary")

                gr.Markdown("---")

                save_btn = gr.Button("💾 Save All Changes", variant="primary")
                export_btn = gr.Button("📤 Export Clean Splits", variant="primary")

                gr.Markdown("---")
                stats_md = gr.Markdown(app.get_dashboard_stats())

            # Middle column - image
            with gr.Column(scale=2):
                image_display = gr.Image(
                    value=app.render_image(),
                    label="Preview",
                    show_label=False,
                    height=600,
                )
                info_md = gr.Markdown(app.get_current_image_info())
                status_text = gr.Textbox(label="Status", interactive=False)

            # Right column - corner sliders
            with gr.Column(scale=1):
                gr.Markdown("### Edit GT Corners")
                gr.Markdown("*Normalized coordinates [0-1]*")

                corners = app.get_current_corners()

                with gr.Group():
                    gr.Markdown("**Top-Left (TL)**")
                    tl_x = gr.Slider(0, 1, value=float(corners[0]), step=0.001, label="X", elem_classes=["corner-slider"])
                    tl_y = gr.Slider(0, 1, value=float(corners[1]), step=0.001, label="Y", elem_classes=["corner-slider"])

                with gr.Group():
                    gr.Markdown("**Top-Right (TR)**")
                    tr_x = gr.Slider(0, 1, value=float(corners[2]), step=0.001, label="X", elem_classes=["corner-slider"])
                    tr_y = gr.Slider(0, 1, value=float(corners[3]), step=0.001, label="Y", elem_classes=["corner-slider"])

                with gr.Group():
                    gr.Markdown("**Bottom-Right (BR)**")
                    br_x = gr.Slider(0, 1, value=float(corners[4]), step=0.001, label="X", elem_classes=["corner-slider"])
                    br_y = gr.Slider(0, 1, value=float(corners[5]), step=0.001, label="Y", elem_classes=["corner-slider"])

                with gr.Group():
                    gr.Markdown("**Bottom-Left (BL)**")
                    bl_x = gr.Slider(0, 1, value=float(corners[6]), step=0.001, label="X", elem_classes=["corner-slider"])
                    bl_y = gr.Slider(0, 1, value=float(corners[7]), step=0.001, label="Y", elem_classes=["corner-slider"])

                apply_btn = gr.Button("✅ Apply Corner Changes", variant="primary")

        # All outputs for refresh
        all_outputs = [
            image_display, info_md, stats_md,
            tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y,
            position_input, total_label, status_text,
        ]

        # Event handlers
        filter_dropdown.change(update_filter, inputs=[filter_dropdown], outputs=all_outputs)
        sort_dropdown.change(update_sort, inputs=[sort_dropdown], outputs=all_outputs)
        prev_btn.click(go_prev, outputs=all_outputs)
        next_btn.click(go_next, outputs=all_outputs)
        position_input.submit(go_to_index, inputs=[position_input], outputs=all_outputs)

        apply_btn.click(
            apply_corners,
            inputs=[tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y],
            outputs=all_outputs,
        )

        discard_btn.click(discard_image, outputs=all_outputs)
        undiscard_btn.click(undiscard_image, outputs=all_outputs)
        undo_btn.click(undo_changes, outputs=all_outputs)
        save_btn.click(save_all, outputs=[stats_md, status_text])
        export_btn.click(export_splits, outputs=[status_text])

        # Keyboard shortcuts
        demo.load(
            lambda: refresh_all(),
            outputs=all_outputs,
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Worst Cases Review Web Application")
    parser.add_argument(
        "--worst_cases",
        type=str,
        default="./evaluation_results/worst_cases.txt",
        help="Path to worst_cases.txt from evaluate_worst_cases.py",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new",
        help="Path to dataset directory with images/ and labels/",
    )
    parser.add_argument(
        "--images_tar",
        type=str,
        default=None,
        help="Path to worst_images.tar.gz for preview images (optional)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory with config.json and best_model.weights.h5 for inference overlay",
    )
    parser.add_argument(
        "--input_norm",
        type=str,
        default="imagenet",
        choices=["imagenet", "zero_one", "raw255"],
        help="Input normalization for inference overlay (default: imagenet or config if present)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Port to run the web app on",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link",
    )
    args = parser.parse_args()

    print(f"Starting Worst Cases Review App...")
    print(f"  Worst cases: {args.worst_cases}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Images tar: {args.images_tar}")
    if args.checkpoint:
        print(f"  Checkpoint: {args.checkpoint}")

    inference_model = None
    inference_img_size = None
    input_norm = args.input_norm

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        config_path = ckpt / "config.json"
        weights_path = ckpt / "best_model.weights.h5"
        if not weights_path.exists():
            alt_weights = ckpt / "final_model.weights.h5"
            if alt_weights.exists():
                weights_path = alt_weights

        if config_path.exists() and weights_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            inference_img_size = int(cfg.get("img_size", 224))
            input_norm = cfg.get("input_norm", input_norm)

            from model import load_inference_model

            inference_model = load_inference_model(
                str(weights_path),
                backbone=cfg.get("backbone", "mobilenetv2"),
                alpha=cfg.get("alpha", 0.35),
                fpn_ch=cfg.get("fpn_ch", 32),
                simcc_ch=cfg.get("simcc_ch", 96),
                img_size=inference_img_size,
                num_bins=cfg.get("num_bins", inference_img_size),
                tau=cfg.get("tau", 1.0),
            )
            print(f"Inference model loaded: {weights_path.name}")
        else:
            print("Warning: checkpoint missing config.json or weights, inference overlay disabled.")

    app = WorstCaseReviewApp(
        worst_cases_path=args.worst_cases,
        dataset_path=args.dataset,
        images_tar_path=args.images_tar,
        inference_model=inference_model,
        inference_img_size=inference_img_size,
        input_norm=input_norm,
    )

    demo = create_app(app)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
