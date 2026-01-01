"""
Collage Review Web Application for DocCornerNet (v2).

Gradio-based interface for fast multi-image review with GT + inference overlays.
Designed for bulk discard decisions with pagination and filters.

Features:
- Collage view with 25/50/100 images per page
- Multi-select and discard images
- Filters and sorting (same as worst-case review)
- GT (green) + inference (red) overlay for each image
- Next/previous page navigation

Usage:
    python web_review2.py \
        --worst_cases /workspace/worst_err_gt20.txt \
        --dataset /workspace/doc-scanner-dataset \
        --checkpoint /workspace/checkpoints/mobilenetv2_256_clean_iter3
"""

import argparse
import json
import math
import sys
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

from model import load_inference_model

# Constants
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp", ".WEBP"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CollageReviewApp:
    def __init__(
        self,
        worst_cases_path: str,
        dataset_path: str,
        checkpoint_path: str,
        display_size: int = 320,
        input_norm: str = "imagenet",
        discarded_file: Optional[str] = None,
    ):
        self.worst_cases_path = Path(worst_cases_path)
        self.dataset_path = Path(dataset_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.display_size = display_size
        self.input_norm = (input_norm or "imagenet").strip().lower()
        self.discarded_path = Path(discarded_file) if discarded_file else (self.dataset_path / "discarded_worst.txt")

        # State
        self.entries: List[Dict] = []
        self.gt_labels: Dict[str, np.ndarray] = {}
        self.pred_cache: Dict[str, np.ndarray] = {}
        self.discarded: set = set()
        self.selected: set = set()

        self.current_filter = "all"
        self.current_sort = "error_desc"
        self.filtered_list: List[str] = []

        # Model
        self.model = None
        self.img_size = 256
        self.config = {}
        self._load_config()
        self._load_model()

        # Load data
        self._load_worst_cases()
        self._load_gt_labels()
        self._load_discarded()
        self._apply_filter_and_sort()

    def _load_config(self):
        config_path = self.checkpoint_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
            self.img_size = int(self.config.get("img_size", 256))
            cfg_norm = str(self.config.get("input_norm", "")).strip().lower()
            if cfg_norm and self.input_norm == "imagenet":
                self.input_norm = cfg_norm

    def _load_model(self):
        weights_path = self.checkpoint_path / "best_model.weights.h5"
        if not weights_path.exists():
            alt_weights = self.checkpoint_path / "final_model.weights.h5"
            if alt_weights.exists():
                weights_path = alt_weights
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing weights: {weights_path}")

        cfg = self.config or {}
        self.model = load_inference_model(
            str(weights_path),
            backbone=cfg.get("backbone", "mobilenetv2"),
            alpha=cfg.get("alpha", 0.35),
            fpn_ch=cfg.get("fpn_ch", 32),
            simcc_ch=cfg.get("simcc_ch", 96),
            img_size=self.img_size,
            num_bins=cfg.get("num_bins", self.img_size),
            tau=cfg.get("tau", 1.0),
        )

    def _load_worst_cases(self):
        if not self.worst_cases_path.exists():
            raise FileNotFoundError(f"Worst cases file not found: {self.worst_cases_path}")

        with open(self.worst_cases_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("=") or line.startswith("Split") or line.startswith("SUMMARY"):
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        self.entries.append({
                            "split": parts[0],
                            "name": parts[1],
                            "iou": float(parts[2]),
                            "err_mean": float(parts[3]),
                            "err_max": float(parts[4]),
                            "score": float(parts[5]) if len(parts) > 5 else 1.0,
                        })
                    except ValueError:
                        continue

        print(f"Loaded {len(self.entries)} entries from {self.worst_cases_path}")

    def _load_gt_labels(self):
        labels_dir = self.dataset_path / "labels"
        if not labels_dir.exists():
            print(f"Warning: labels directory not found: {labels_dir}")
            return

        for entry in self.entries:
            name = entry["name"]
            stem = Path(name).stem
            label_path = labels_dir / f"{stem}.txt"
            if label_path.exists():
                try:
                    with open(label_path, "r") as f:
                        line = f.readline().strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 9 and parts[0] == "0":
                                coords = np.array([float(parts[i]) for i in range(1, 9)], dtype=np.float32)
                                self.gt_labels[name] = coords
                except Exception as exc:
                    print(f"Warning: failed to load label for {name}: {exc}")

        print(f"Loaded {len(self.gt_labels)} GT labels")

    def _load_discarded(self):
        if self.discarded_path.exists():
            with open(self.discarded_path, "r") as f:
                for line in f:
                    name = line.strip()
                    if name:
                        self.discarded.add(name)
            print(f"Loaded {len(self.discarded)} discarded images")

    def _save_discarded(self):
        self.discarded_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.discarded_path, "w") as f:
            for name in sorted(self.discarded):
                f.write(f"{name}\n")

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
        if name in self.pred_cache:
            return self.pred_cache[name]

        img_path = self._resolve_image_path(name)
        if img_path is None:
            return None

        img = cv2.imread(str(img_path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        x = self._normalize_for_model(img)
        x = x[None, ...]

        preds = self.model(x, training=False)
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

    def _draw_polygon(self, img: np.ndarray, coords: np.ndarray, color: Tuple[int, int, int]):
        if coords is None or len(coords) != 8:
            return
        h, w = img.shape[:2]
        pts = coords.reshape(4, 2)
        pts = np.stack([pts[:, 0] * w, pts[:, 1] * h], axis=1).astype(np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        for p in pts:
            cv2.circle(img, tuple(p), 3, color, -1)

    def _render_entry(self, name: str, entry: Dict, selected: bool = False, size: Optional[int] = None) -> Image.Image:
        render_size = int(size or self.display_size)
        img_path = self._resolve_image_path(name)
        if img_path is None:
            placeholder = np.zeros((render_size, render_size, 3), dtype=np.uint8)
            return Image.fromarray(placeholder)

        img = cv2.imread(str(img_path))
        if img is None:
            placeholder = np.zeros((render_size, render_size, 3), dtype=np.uint8)
            return Image.fromarray(placeholder)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (render_size, render_size))

        gt = self.gt_labels.get(name)
        pred = self._get_pred_coords(name)

        self._draw_polygon(img, gt, (0, 255, 0))   # GT green
        self._draw_polygon(img, pred, (255, 0, 0))  # Pred red

        if selected:
            # Draw selection badge (green box with check)
            cv2.rectangle(img, (4, 4), (36, 36), (0, 180, 0), -1)
            cv2.putText(img, "✓", (9, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return Image.fromarray(img)

    def _apply_filter_and_sort(self):
        if self.current_filter == "all":
            filtered = [e["name"] for e in self.entries if e["name"] not in self.discarded]
        elif self.current_filter == "train":
            filtered = [e["name"] for e in self.entries
                        if e["split"] == "train" and e["name"] not in self.discarded]
        elif self.current_filter == "val":
            filtered = [e["name"] for e in self.entries
                        if e["split"] == "val" and e["name"] not in self.discarded]
        elif self.current_filter == "discarded":
            filtered = [e["name"] for e in self.entries if e["name"] in self.discarded]
        elif self.current_filter == "high_error":
            filtered = [e["name"] for e in self.entries
                        if e["err_mean"] > 20 and e["name"] not in self.discarded]
        else:
            filtered = [e["name"] for e in self.entries if e["name"] not in self.discarded]

        entry_map = {e["name"]: e for e in self.entries}
        if self.current_sort == "error_desc":
            filtered.sort(key=lambda n: -entry_map[n]["err_mean"])
        elif self.current_sort == "error_asc":
            filtered.sort(key=lambda n: entry_map[n]["err_mean"])
        elif self.current_sort == "iou_asc":
            filtered.sort(key=lambda n: entry_map[n]["iou"])
        elif self.current_sort == "iou_desc":
            filtered.sort(key=lambda n: -entry_map[n]["iou"])
        elif self.current_sort == "name":
            filtered.sort()

        self.filtered_list = filtered

    def get_page(self, page_index: int, per_page: int, selected: Optional[set] = None):
        total = len(self.filtered_list)
        total_pages = max(1, math.ceil(total / per_page)) if total else 1
        page_index = max(0, min(page_index, total_pages - 1))
        start = page_index * per_page
        end = min(start + per_page, total)
        names = self.filtered_list[start:end]
        selected = selected or set()

        entry_map = {e["name"]: e for e in self.entries}
        gallery_items = []
        for name in names:
            entry = entry_map.get(name, {})
            img = self._render_entry(name, entry, selected=name in selected)
            caption = f"iou={entry.get('iou', 0):.3f} | err={entry.get('err_mean', 0):.1f}"
            gallery_items.append((img, caption))

        if total == 0:
            info = "Page 1/1 | Showing 0-0 of 0"
        else:
            info = f"Page {page_index + 1}/{total_pages} | Showing {start + 1}-{end} of {total}"
        return gallery_items, names, info, page_index

    def discard(self, selected_names: List[str]) -> int:
        if not selected_names:
            return 0
        added = 0
        for name in selected_names:
            if name and name not in self.discarded:
                self.discarded.add(name)
                added += 1
        if added:
            self._save_discarded()
            self._apply_filter_and_sort()
        return added


def build_ui(app: CollageReviewApp, host: str, port: int):
    def refresh(page_number, per_page, filter_choice, sort_choice):
        try:
            page_number = int(page_number)
        except (TypeError, ValueError):
            page_number = 1
        page_number = max(1, page_number)
        page_index = page_number - 1
        app.current_filter = filter_choice
        app.current_sort = sort_choice
        app._apply_filter_and_sort()
        gallery, names, info, page_index = app.get_page(page_index, per_page, app.selected)
        selected_count = f"Selected: {len(app.selected)}"
        return gallery, names, info, page_index + 1, selected_count

    def go_prev(page_number, per_page, filter_choice, sort_choice):
        try:
            page_number = int(page_number)
        except (TypeError, ValueError):
            page_number = 1
        return refresh(max(1, page_number - 1), per_page, filter_choice, sort_choice)

    def go_next(page_number, per_page, filter_choice, sort_choice):
        try:
            page_number = int(page_number)
        except (TypeError, ValueError):
            page_number = 1
        return refresh(page_number + 1, per_page, filter_choice, sort_choice)

    def discard_selected(page_number, per_page, filter_choice, sort_choice):
        selected_names = list(app.selected)
        count = app.discard(selected_names)
        for name in list(app.selected):
            if name in app.discarded:
                app.selected.discard(name)
        gallery, names, info, page_num, selected_count = refresh(page_number, per_page, filter_choice, sort_choice)
        status = f"Discarded {count} images." if count else "No images discarded."
        return gallery, names, info, page_num, selected_count, status

    def clear_selected(page_number, per_page, filter_choice, sort_choice):
        app.selected.clear()
        gallery, names, info, page_num, selected_count = refresh(page_number, per_page, filter_choice, sort_choice)
        return gallery, names, info, page_num, selected_count, "Selection cleared."

    def toggle_select(evt: gr.SelectData, page_names, page_number, per_page, filter_choice, sort_choice):
        if evt is None or evt.index is None:
            return None, None, None, None
        try:
            idx = int(evt.index)
        except Exception:
            return None, None, None, None
        if idx < 0 or idx >= len(page_names):
            return None, None, None, None
        name = page_names[idx]
        if name in app.selected:
            app.selected.remove(name)
        else:
            app.selected.add(name)
        entry_map = {e["name"]: e for e in app.entries}
        preview = app._render_entry(name, entry_map.get(name, {}), selected=True, size=max(512, app.display_size))
        gallery, names, info, page_num, selected_count = refresh(page_number, per_page, filter_choice, sort_choice)
        return gallery, names, selected_count, preview

    with gr.Blocks(title="DocCornerNet Review v2") as demo:
        gr.Markdown("## DocCornerNet Web Review v2")
        gr.Markdown("Green = GT, Red = Inference. Click images to toggle selection; click again to unselect.")

        with gr.Row():
            filter_dropdown = gr.Dropdown(
                choices=["all", "train", "val", "high_error", "discarded"],
                value="all",
                label="Filter",
            )
            sort_dropdown = gr.Dropdown(
                choices=["error_desc", "error_asc", "iou_desc", "iou_asc", "name"],
                value="error_desc",
                label="Sort",
            )
            per_page = gr.Dropdown(
                choices=[25, 50, 100],
                value=100,
                label="Images per page",
            )
            page_number = gr.Number(value=1, precision=0, label="Page")
            selected_label = gr.Markdown("Selected: 0")

        with gr.Row():
            prev_btn = gr.Button("Prev")
            next_btn = gr.Button("Next")
            page_info = gr.Markdown("Page 1/1 | Showing 0-0 of 0")

        gallery = gr.Gallery(
            label="Collage",
            columns=10,
            height=800,
            show_label=False,
            preview=False,
        )

        with gr.Row():
            preview = gr.Image(label="Preview", height=520)

        with gr.Row():
            discard_btn = gr.Button("Discard selected", variant="primary")
            clear_btn = gr.Button("Clear selection")
            status = gr.Markdown("")

        page_names_state = gr.State([])

        demo.load(
            refresh,
            inputs=[page_number, per_page, filter_dropdown, sort_dropdown],
            outputs=[gallery, page_names_state, page_info, page_number, selected_label],
        )

        filter_dropdown.change(
            refresh,
            inputs=[page_number, per_page, filter_dropdown, sort_dropdown],
            outputs=[gallery, page_names_state, page_info, page_number, selected_label],
        )
        sort_dropdown.change(
            refresh,
            inputs=[page_number, per_page, filter_dropdown, sort_dropdown],
            outputs=[gallery, page_names_state, page_info, page_number, selected_label],
        )
        per_page.change(
            refresh,
            inputs=[page_number, per_page, filter_dropdown, sort_dropdown],
            outputs=[gallery, page_names_state, page_info, page_number, selected_label],
        )

        prev_btn.click(
            go_prev,
            inputs=[page_number, per_page, filter_dropdown, sort_dropdown],
            outputs=[gallery, page_names_state, page_info, page_number, selected_label],
        )
        next_btn.click(
            go_next,
            inputs=[page_number, per_page, filter_dropdown, sort_dropdown],
            outputs=[gallery, page_names_state, page_info, page_number, selected_label],
        )
        page_number.change(
            refresh,
            inputs=[page_number, per_page, filter_dropdown, sort_dropdown],
            outputs=[gallery, page_names_state, page_info, page_number, selected_label],
        )

        gallery.select(
            toggle_select,
            inputs=[page_names_state, page_number, per_page, filter_dropdown, sort_dropdown],
            outputs=[gallery, page_names_state, selected_label, preview],
        )

        discard_btn.click(
            discard_selected,
            inputs=[page_number, per_page, filter_dropdown, sort_dropdown],
            outputs=[gallery, page_names_state, page_info, page_number, selected_label, status],
        )
        clear_btn.click(
            clear_selected,
            inputs=[page_number, per_page, filter_dropdown, sort_dropdown],
            outputs=[gallery, page_names_state, page_info, page_number, selected_label, status],
        )

    demo.launch(server_name=host, server_port=port)


def main():
    parser = argparse.ArgumentParser(description="DocCornerNet web review v2 (collage).")
    parser.add_argument("--worst_cases", type=str, required=True, help="Path to worst_cases.txt")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset root")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint dir")
    parser.add_argument("--display_size", type=int, default=320, help="Display size for each image")
    parser.add_argument("--input_norm", type=str, default="imagenet", help="imagenet|zero_one|raw255")
    parser.add_argument("--discarded_file", type=str, default=None, help="Optional discarded file path")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app = CollageReviewApp(
        worst_cases_path=args.worst_cases,
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        display_size=args.display_size,
        input_norm=args.input_norm,
        discarded_file=args.discarded_file,
    )
    build_ui(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
