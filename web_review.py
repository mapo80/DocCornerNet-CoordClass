"""
Outlier Review Web Application for DocCornerNet.

Gradio-based interface for reviewing model predictions vs ground truth,
allowing users to identify outliers, modify GT annotations, and discard
bad samples from the dataset.

Features:
- View GT bbox (green) and model prediction (red) for each image
- Model quality dashboard with aggregate metrics
- Filter by IoU/error thresholds to identify outliers
- Modify GT annotations via sliders or click-to-use-prediction
- Discard images from dataset
- Export clean split file without discarded images

Usage:
    python web_review.py --dataset /path/to/dataset --checkpoint checkpoints/model --split val
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Defer TensorFlow import for faster startup message
print("Loading dependencies...")

import tensorflow as tf

# Disable GPU for review app (runs on CPU for simplicity)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    import gradio as gr
except ImportError:
    print("Error: gradio not installed. Run: pip install gradio")
    sys.exit(1)

from model import load_inference_model
from metrics import compute_polygon_iou, compute_corner_error
from dataset import load_split_file, load_yolo_label

# Constants
CORNER_LABELS = ["TL", "TR", "BR", "BL"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


CORNER_CLICK_RADIUS = 20  # Pixels - radius for corner selection
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp", ".WEBP"]
IMAGE_QUARANTINE_DIRNAME = "image-quarantine"
LABEL_QUARANTINE_DIRNAME = "label-quarantine"
DISCARDED_FILENAME = "discarded.txt"
DEFAULT_CANDIDATE_CHECKPOINT = "checkpoints/mobilenetv2_256_best"
DEFAULT_TEACHER_ONNX = "teacher/fastvit_sa24_h_e_bifpn_256_fp32.onnx"
DEFAULT_TEACHER_INPUT_NORM = "zero_one"
DEFAULT_PURPLE_CHECKPOINT = "checkpoints/mobilenetv2_224_iou98"


class OutlierReviewApp:
    """
    Web application for reviewing model predictions and GT annotations.
    """

    def __init__(
        self,
        dataset_path: str,
        checkpoint_path: str,
        split: str = "val",
        batch_size: int = 32,
        candidate_checkpoint_path: Optional[str] = None,
        teacher_onnx_path: Optional[str] = None,
        teacher_input_norm: str = DEFAULT_TEACHER_INPUT_NORM,
        purple_checkpoint_path: Optional[str] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.split = split
        self.batch_size = batch_size
        self.candidate_checkpoint_path = Path(
            candidate_checkpoint_path or DEFAULT_CANDIDATE_CHECKPOINT
        )
        self.teacher_onnx_path = Path(teacher_onnx_path or DEFAULT_TEACHER_ONNX)
        self.teacher_input_norm = (teacher_input_norm or DEFAULT_TEACHER_INPUT_NORM).strip().lower()
        self.purple_checkpoint_path = Path(purple_checkpoint_path or DEFAULT_PURPLE_CHECKPOINT)

        self.image_quarantine_dir = self.dataset_path / IMAGE_QUARANTINE_DIRNAME
        self.label_quarantine_dir = self.dataset_path / LABEL_QUARANTINE_DIRNAME

        # State
        self.image_list: List[str] = []
        self.predictions: Dict[str, np.ndarray] = {}
        self.gt_labels: Dict[str, np.ndarray] = {}
        self.scores: Dict[str, float] = {}
        self.metrics: Dict[str, Dict] = {}
        self.modified: Dict[str, np.ndarray] = {}  # Modified GT coords
        self.proposals: Dict[str, np.ndarray] = {}  # Proposed bbox (pending approval)
        self.discarded: set = set()  # Discarded image names

        self.current_index = 0
        self.current_filter = "all"
        self.current_sort = "error_desc"
        self.filtered_list: List[str] = []
        self.status_message: str = ""

        # Drag state
        self.selected_corner: Optional[int] = None  # 0-3 for TL, TR, BR, BL

        # Candidate checkpoint (used for proposals)
        self.candidate_model = None
        self.candidate_config = None
        self.candidate_img_size = 256
        self.candidate_predictions: Dict[str, np.ndarray] = {}
        self.candidate_scores: Dict[str, float] = {}
        self._candidate_cache_attempted = False

        # Purple overlay checkpoint (always-on overlay bbox)
        self.purple_model = None
        self.purple_config = None
        self.purple_img_size = 224
        self.purple_predictions: Dict[str, np.ndarray] = {}
        self._purple_cache_attempted = False

        # ONNX teacher (used for proposals)
        self.teacher_session = None
        self.teacher_input_name: Optional[str] = None
        self.teacher_output_name: Optional[str] = None
        self.teacher_img_size = 256
        self.teacher_predictions: Dict[str, np.ndarray] = {}
        self._teacher_cache_attempted = False

        # GT geometry filters
        self.gt_acute_angle_threshold_deg: float = 60.0

        # Model config
        self.img_size = 224  # Model input size
        self.display_size = 512  # Display size for review UI
        self.model = None
        self.config = None

        # Load everything
        self._load_config()
        self._load_split()
        self._load_discarded()

        # Try to load cached predictions, otherwise run inference
        if not self._load_predictions_cache():
            self._load_model()
            self._run_inference()
            self._save_predictions_cache()

        self._compute_metrics()
        self._ensure_purple_predictions_for_overlay()
        self._compute_purple_metrics()
        self._apply_filter_and_sort()

    def _load_config(self):
        """Load model configuration from checkpoint."""
        config_path = self.checkpoint_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
            self.img_size = self.config.get("img_size", 224)
        else:
            print(f"Warning: config.json not found in {self.checkpoint_path}")
            self.config = {
                "backbone": "mobilenetv2",
                "alpha": 0.35,
                "img_size": 224,
                "num_bins": 224,
                "fpn_ch": 32,
                "simcc_ch": 96,
                "tau": 1.0,
            }

    def _load_split(self):
        """Load image list from split file."""
        # Try different split file formats
        split_files = [
            self.dataset_path / f"{self.split}_with_negative_v2.txt",
            self.dataset_path / f"{self.split}_with_negative.txt",
            self.dataset_path / f"{self.split}.txt",
        ]

        for split_file in split_files:
            if split_file.exists():
                self.image_list = load_split_file(str(split_file))
                print(f"Loaded {len(self.image_list)} images from {split_file.name}")
                break
        else:
            raise FileNotFoundError(f"No split file found for '{self.split}'")

        # Load GT labels
        for name in self.image_list:
            if name.startswith("negative_"):
                self.gt_labels[name] = np.zeros(8, dtype=np.float32)
            else:
                label_path = self._locate_label_path(name)
                if label_path is None:
                    self.gt_labels[name] = np.zeros(8, dtype=np.float32)
                else:
                    coords, _has_doc = load_yolo_label(str(label_path))
                    self.gt_labels[name] = coords

    def _load_discarded(self):
        """Load list of discarded images."""
        discarded_path = self.dataset_path / DISCARDED_FILENAME
        if discarded_path.exists():
            with open(discarded_path) as f:
                for line in f:
                    name = line.strip()
                    if name:
                        self.discarded.add(name)
            print(f"Loaded {len(self.discarded)} discarded images")

    def _get_cache_path(self) -> Path:
        """Get path to predictions cache file."""
        # Cache file is named after checkpoint and split
        checkpoint_name = self.checkpoint_path.name
        cache_name = f".predictions_cache_{checkpoint_name}_{self.split}.npz"
        return self.dataset_path / cache_name

    def _load_predictions_cache(self) -> bool:
        """Load predictions from cache if available and valid."""
        cache_path = self._get_cache_path()

        if not cache_path.exists():
            print(f"No predictions cache found at {cache_path}")
            return False

        try:
            print(f"Loading predictions from cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)

            # Verify cache matches current image list
            cached_names = list(data["names"])
            if cached_names != self.image_list:
                print("Cache is stale (image list changed), will recompute")
                return False

            # Load predictions and scores
            predictions = data["predictions"]
            scores = data["scores"]

            for i, name in enumerate(self.image_list):
                self.predictions[name] = predictions[i]
                self.scores[name] = float(scores[i])

            print(f"Loaded {len(self.predictions)} cached predictions")
            return True

        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False

    def _save_predictions_cache(self):
        """Save predictions to cache file."""
        cache_path = self._get_cache_path()

        try:
            # Prepare arrays
            names = np.array(self.image_list, dtype=object)
            predictions = np.array([self.predictions[n] for n in self.image_list])
            scores = np.array([self.scores[n] for n in self.image_list])

            np.savez(cache_path, names=names, predictions=predictions, scores=scores)
            print(f"Saved predictions cache to {cache_path}")

        except Exception as e:
            print(f"Failed to save cache: {e}")

    def _load_model(self):
        """Load inference model from checkpoint."""
        print("Loading model...")

        weights_path = self.checkpoint_path / "best_model.weights.h5"
        if not weights_path.exists():
            weights_path = self.checkpoint_path / "best_model.keras"
        if not weights_path.exists():
            raise FileNotFoundError(f"No weights found in {self.checkpoint_path}")

        self.model = load_inference_model(
            weights_path=str(weights_path),
            backbone=self.config.get("backbone", "mobilenetv2"),
            alpha=self.config.get("alpha", 0.35),
            img_size=self.config.get("img_size", 224),
            num_bins=self.config.get("num_bins", 224),
            fpn_ch=self.config.get("fpn_ch", 32),
            simcc_ch=self.config.get("simcc_ch", 96),
            tau=self.config.get("tau", 1.0),
        )
        print(f"Model loaded: {self.model.name}")

    # === Candidate checkpoint (for proposals) ===

    def _load_candidate_config(self) -> None:
        if self.candidate_config is not None:
            return
        config_path = self.candidate_checkpoint_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.candidate_config = json.load(f)
            self.candidate_img_size = int(self.candidate_config.get("img_size", 256))
        else:
            self.candidate_config = {
                "backbone": "mobilenetv2",
                "alpha": 0.35,
                "img_size": 256,
                "num_bins": 256,
                "fpn_ch": 32,
                "simcc_ch": 96,
                "tau": 1.0,
            }
            self.candidate_img_size = 256

    def _get_candidate_cache_path(self) -> Path:
        checkpoint_name = self.candidate_checkpoint_path.name
        cache_name = f".predictions_cache_{checkpoint_name}_{self.split}.npz"
        return self.dataset_path / cache_name

    def _load_candidate_predictions_cache(self) -> bool:
        cache_path = self._get_candidate_cache_path()
        if not cache_path.exists():
            return False

        try:
            print(f"Loading candidate predictions from cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            cached_names = list(data["names"])
            if cached_names != self.image_list:
                print("Candidate cache is stale (image list changed), will ignore")
                return False

            predictions = data["predictions"]
            scores = data["scores"]
            for i, name in enumerate(self.image_list):
                self.candidate_predictions[name] = predictions[i]
                self.candidate_scores[name] = float(scores[i])
            return True
        except Exception as e:
            print(f"Failed to load candidate cache: {e}")
            return False

    def _load_candidate_model(self) -> bool:
        if self.candidate_model is not None:
            return True

        if not self.candidate_checkpoint_path.exists():
            self.status_message = f"Candidate checkpoint not found: {self.candidate_checkpoint_path}"
            return False

        self._load_candidate_config()

        weights_path = self.candidate_checkpoint_path / "best_model.weights.h5"
        if not weights_path.exists():
            weights_path = self.candidate_checkpoint_path / "best_model.keras"
        if not weights_path.exists():
            self.status_message = f"No weights found in {self.candidate_checkpoint_path}"
            return False

        print(f"Loading candidate model from {self.candidate_checkpoint_path} ...")
        self.candidate_model = load_inference_model(
            weights_path=str(weights_path),
            backbone=self.candidate_config.get("backbone", "mobilenetv2"),
            alpha=self.candidate_config.get("alpha", 0.35),
            img_size=self.candidate_config.get("img_size", 256),
            num_bins=self.candidate_config.get("num_bins", 256),
            fpn_ch=self.candidate_config.get("fpn_ch", 32),
            simcc_ch=self.candidate_config.get("simcc_ch", 96),
            tau=self.candidate_config.get("tau", 1.0),
        )
        return True

    def _predict_candidate_single(self, name: str) -> Optional[np.ndarray]:
        if not self._load_candidate_model():
            return None

        img_size = int(self.candidate_img_size or 256)
        img = self._load_image_for_inference(name, img_size)
        coords, score_logits = self.candidate_model.predict(img[None, ...], verbose=0)

        logit = float(score_logits.flatten()[0])
        score = 1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))

        self.candidate_predictions[name] = coords[0]
        self.candidate_scores[name] = float(score)
        return coords[0]

    def get_candidate_prediction(self, name: str) -> Optional[np.ndarray]:
        """Get candidate checkpoint prediction for a sample name."""
        if name in self.candidate_predictions:
            return self.candidate_predictions[name]

        if not self._candidate_cache_attempted:
            self._candidate_cache_attempted = True
            self._load_candidate_predictions_cache()

        if name in self.candidate_predictions:
            return self.candidate_predictions[name]

        return self._predict_candidate_single(name)

    # === Purple checkpoint (overlay bbox) ===

    def _load_purple_config(self) -> None:
        if self.purple_config is not None:
            return
        config_path = self.purple_checkpoint_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.purple_config = json.load(f)
            self.purple_img_size = int(self.purple_config.get("img_size", 224))
        else:
            self.purple_config = {
                "backbone": "mobilenetv2",
                "alpha": 0.35,
                "img_size": 224,
                "num_bins": 224,
                "fpn_ch": 32,
                "simcc_ch": 96,
                "tau": 1.0,
            }
            self.purple_img_size = 224

    def _get_purple_cache_paths(self) -> List[Path]:
        checkpoint_name = self.purple_checkpoint_path.name
        return [
            self.dataset_path / f".predictions_cache_purple_{checkpoint_name}_{self.split}.npz",
            # Fallback to the standard cache format if it already exists.
            self.dataset_path / f".predictions_cache_{checkpoint_name}_{self.split}.npz",
        ]

    def _load_purple_predictions_cache(self) -> bool:
        for cache_path in self._get_purple_cache_paths():
            if not cache_path.exists():
                continue
            try:
                print(f"Loading purple predictions from cache: {cache_path}")
                data = np.load(cache_path, allow_pickle=True)
                cached_names = list(data["names"])
                if cached_names != self.image_list:
                    print("Purple cache is stale (image list changed), will ignore")
                    continue
                predictions = data["predictions"]
                for i, name in enumerate(self.image_list):
                    self.purple_predictions[name] = predictions[i]
                return True
            except Exception as e:
                print(f"Failed to load purple cache: {e}")
                continue
        return False

    def _save_purple_predictions_cache(self, predictions: np.ndarray) -> None:
        cache_path = self._get_purple_cache_paths()[0]
        try:
            names = np.array(self.image_list, dtype=object)
            np.savez(
                cache_path,
                names=names,
                predictions=np.asarray(predictions, dtype=np.float32),
                checkpoint=str(self.purple_checkpoint_path),
                img_size=int(self.purple_img_size or 0),
            )
            print(f"Saved purple predictions cache to {cache_path}")
        except Exception as e:
            print(f"Failed to save purple cache: {e}")

    def _load_purple_model(self) -> bool:
        if self.purple_model is not None:
            return True
        if not self.purple_checkpoint_path.exists():
            print(f"Purple checkpoint not found: {self.purple_checkpoint_path}")
            return False

        self._load_purple_config()

        weights_path = self.purple_checkpoint_path / "best_model.weights.h5"
        if not weights_path.exists():
            weights_path = self.purple_checkpoint_path / "best_model.keras"
        if not weights_path.exists():
            weights_path = self.purple_checkpoint_path / "best_model_inference.keras"
        if not weights_path.exists():
            print(f"No weights found in {self.purple_checkpoint_path}")
            return False

        print(f"Loading purple model from {self.purple_checkpoint_path} ...")
        self.purple_model = load_inference_model(
            weights_path=str(weights_path),
            backbone=self.purple_config.get("backbone", "mobilenetv2"),
            alpha=self.purple_config.get("alpha", 0.35),
            img_size=self.purple_config.get("img_size", 224),
            num_bins=self.purple_config.get("num_bins", 224),
            fpn_ch=self.purple_config.get("fpn_ch", 32),
            simcc_ch=self.purple_config.get("simcc_ch", 96),
            tau=self.purple_config.get("tau", 1.0),
        )
        return True

    def _predict_purple_single(self, name: str) -> Optional[np.ndarray]:
        if not self._load_purple_model():
            return None
        img_size = int(self.purple_img_size or 224)
        img = self._load_image_for_inference(name, img_size)
        coords, _score_logits = self.purple_model.predict(img[None, ...], verbose=0)
        self.purple_predictions[name] = coords[0]
        return coords[0]

    def get_purple_prediction(self, name: str) -> Optional[np.ndarray]:
        if name in self.purple_predictions:
            return self.purple_predictions[name]
        if not self._purple_cache_attempted:
            self._purple_cache_attempted = True
            self._load_purple_predictions_cache()
        if name in self.purple_predictions:
            return self.purple_predictions[name]
        return self._predict_purple_single(name)

    def _precompute_purple_predictions(self, batch_size: int = 32) -> None:
        if not self._load_purple_model():
            return

        img_size = int(self.purple_img_size or 224)
        n_total = len(self.image_list)
        preds = np.zeros((n_total, 8), dtype=np.float32)

        batch_imgs: List[np.ndarray] = []
        batch_indices: List[int] = []
        processed = 0
        missing = 0

        n_nonneg = 0
        for n in self.image_list:
            if not n.startswith("negative_"):
                n_nonneg += 1

        print(f"Precomputing purple bboxes ({self.purple_checkpoint_path.name}) for {n_nonneg} non-negative samples...")

        def flush_batch():
            nonlocal batch_imgs, batch_indices
            if not batch_imgs:
                return
            batch = np.stack(batch_imgs, axis=0)
            coords, _score_logits = self.purple_model.predict(batch, verbose=0)
            for j, idx in enumerate(batch_indices):
                name_j = self.image_list[idx]
                coords_j = np.asarray(coords[j], dtype=np.float32)
                preds[idx] = coords_j
                self.purple_predictions[name_j] = coords_j
            batch_imgs = []
            batch_indices = []

        for idx, name in enumerate(self.image_list):
            if name.startswith("negative_"):
                continue

            img_path = self._locate_image_path(name, include_quarantine=True)
            if img_path is None or not img_path.exists():
                missing += 1
                continue

            img = self._load_image_for_inference(name, img_size)
            batch_imgs.append(img)
            batch_indices.append(idx)

            if len(batch_imgs) >= batch_size:
                flush_batch()

            processed += 1
            if processed % 500 == 0:
                print(f"  Purple processed {processed}/{n_nonneg} samples...")

        flush_batch()

        self._save_purple_predictions_cache(preds)
        if missing:
            print(f"Purple precompute complete (missing/unreadable images: {missing}).")
        else:
            print("Purple precompute complete.")

    def _ensure_purple_predictions_for_overlay(self) -> None:
        if not self.purple_checkpoint_path.exists():
            print(f"Purple checkpoint not found at {self.purple_checkpoint_path}; skipping purple overlay.")
            return
        if self._load_purple_predictions_cache():
            return
        self._precompute_purple_predictions(batch_size=min(int(self.batch_size or 32), 64))

    # === ONNX teacher (for proposals) ===

    def _teacher_convert_to_input(self, image_raw255_nhwc: np.ndarray) -> np.ndarray:
        norm = (self.teacher_input_norm or DEFAULT_TEACHER_INPUT_NORM).strip().lower()
        x = image_raw255_nhwc.astype(np.float32, copy=False)
        if norm == "raw255":
            pass
        elif norm == "zero_one":
            x = x / 255.0
        elif norm == "imagenet":
            x = x / 255.0
            x = (x - IMAGENET_MEAN[None, None, None, :]) / IMAGENET_STD[None, None, None, :]
        elif norm == "m1p1":
            x = (x / 127.5) - 1.0
        else:
            raise ValueError(f"Unsupported teacher_input_norm='{self.teacher_input_norm}'")

        # NCHW
        return np.transpose(x, (0, 3, 1, 2))

    @staticmethod
    def _decode_teacher_soft(heat: np.ndarray) -> np.ndarray:
        """
        heat: [B, 4, h, w]
        returns coords: [B, 8] in [0,1]
        """
        b, c, h, w = heat.shape
        if c != 4:
            raise ValueError(f"Expected 4 channels, got {c}")

        eps = 1e-9
        x_m = heat.sum(axis=2)  # [B,4,w]
        y_m = heat.sum(axis=3)  # [B,4,h]

        xs = np.arange(w, dtype=np.float32)[None, None, :]
        ys = np.arange(h, dtype=np.float32)[None, None, :]

        x_sum = x_m.sum(axis=-1, keepdims=True) + eps  # [B,4,1]
        y_sum = y_m.sum(axis=-1, keepdims=True) + eps

        x_px = (x_m * xs).sum(axis=-1) / x_sum.squeeze(-1)  # [B,4]
        y_px = (y_m * ys).sum(axis=-1) / y_sum.squeeze(-1)  # [B,4]

        x01 = x_px / float(max(w - 1, 1))
        y01 = y_px / float(max(h - 1, 1))
        coords = np.stack([x01, y01], axis=-1).reshape(b, 8).astype(np.float32)
        return np.clip(coords, 0.0, 1.0)

    @staticmethod
    def _order_quad_xy(pts_xy: np.ndarray) -> np.ndarray:
        """Reorder 4 points to (TL, TR, BR, BL) using sum/diff heuristic."""
        s = pts_xy.sum(axis=1)
        d = pts_xy[:, 0] - pts_xy[:, 1]
        tl = pts_xy[int(np.argmin(s))]
        br = pts_xy[int(np.argmax(s))]
        tr = pts_xy[int(np.argmax(d))]
        bl = pts_xy[int(np.argmin(d))]
        return np.stack([tl, tr, br, bl], axis=0)

    @classmethod
    def _order_quads(cls, coords_8: np.ndarray) -> np.ndarray:
        """coords_8: [B,8] -> reordered [B,8] (TL,TR,BR,BL)."""
        b = coords_8.shape[0]
        out = np.empty_like(coords_8, dtype=np.float32)
        for i in range(b):
            pts = coords_8[i].reshape(4, 2)
            out[i] = cls._order_quad_xy(pts).reshape(-1)
        return out

    def _get_teacher_cache_paths(self) -> List[Path]:
        model_name = self.teacher_onnx_path.stem
        norm = (self.teacher_input_norm or DEFAULT_TEACHER_INPUT_NORM).strip().lower()
        # Norm-specific cache (preferred) + legacy cache for backwards compatibility.
        return [
            self.dataset_path / f".predictions_cache_{model_name}_{norm}_{self.split}.npz",
            self.dataset_path / f".predictions_cache_{model_name}_{self.split}.npz",
        ]

    def _load_teacher_predictions_cache(self) -> bool:
        for cache_path in self._get_teacher_cache_paths():
            if not cache_path.exists():
                continue
            try:
                print(f"Loading teacher predictions from cache: {cache_path}")
                data = np.load(cache_path, allow_pickle=True)
                cached_names = list(data["names"])
                if cached_names != self.image_list:
                    print("Teacher cache is stale (image list changed), will ignore")
                    continue
                predictions = data["predictions"]
                for i, name in enumerate(self.image_list):
                    self.teacher_predictions[name] = predictions[i]
                return True
            except Exception as e:
                print(f"Failed to load teacher cache: {e}")
                continue
        return False

    def _save_teacher_predictions_cache(self, predictions: np.ndarray) -> None:
        cache_path = self._get_teacher_cache_paths()[0]
        try:
            names = np.array(self.image_list, dtype=object)
            np.savez(
                cache_path,
                names=names,
                predictions=np.asarray(predictions, dtype=np.float32),
                teacher_onnx=str(self.teacher_onnx_path),
                teacher_input_norm=str(self.teacher_input_norm),
                teacher_img_size=int(self.teacher_img_size or 0),
            )
            print(f"Saved teacher predictions cache to {cache_path}")
        except Exception as e:
            print(f"Failed to save teacher cache: {e}")

    def _load_teacher_session(self) -> bool:
        if self.teacher_session is not None:
            return True
        if not self.teacher_onnx_path.exists():
            self.status_message = f"Teacher ONNX not found: {self.teacher_onnx_path}"
            return False
        try:
            import onnxruntime as ort
        except Exception as e:
            self.status_message = "onnxruntime is required for teacher proposals"
            print(f"Failed to import onnxruntime: {e}")
            return False

        providers = ["CPUExecutionProvider"]
        self.teacher_session = ort.InferenceSession(str(self.teacher_onnx_path), providers=providers)
        self.teacher_input_name = self.teacher_session.get_inputs()[0].name
        self.teacher_output_name = self.teacher_session.get_outputs()[0].name

        # Infer input size if present
        in_shape = self.teacher_session.get_inputs()[0].shape
        if isinstance(in_shape, (list, tuple)) and len(in_shape) == 4 and isinstance(in_shape[2], int):
            self.teacher_img_size = int(in_shape[2])
        return True

    def _precompute_teacher_predictions(self, batch_size: int = 32) -> None:
        if not self._load_teacher_session():
            return

        img_size = int(self.teacher_img_size or 256)

        # Respect fixed batch dimension if present (common for some ONNX exports).
        in_shape = self.teacher_session.get_inputs()[0].shape
        if isinstance(in_shape, (list, tuple)) and len(in_shape) >= 1 and isinstance(in_shape[0], int):
            batch_size = int(max(1, min(batch_size, in_shape[0])))

        n_total = len(self.image_list)
        preds = np.zeros((n_total, 8), dtype=np.float32)

        batch_imgs: List[np.ndarray] = []
        batch_indices: List[int] = []
        missing = 0
        processed = 0

        print(
            f"Precomputing teacher bboxes ({self.teacher_onnx_path.name}, norm={self.teacher_input_norm}) "
            f"for {sum(1 for n in self.image_list if not n.startswith('negative_'))} non-negative samples..."
        )

        def flush_batch():
            nonlocal batch_imgs, batch_indices
            if not batch_imgs:
                return
            batch = np.stack(batch_imgs, axis=0)  # NHWC raw255
            x = self._teacher_convert_to_input(batch)
            heat = self.teacher_session.run(
                [self.teacher_output_name],
                {self.teacher_input_name: x},
            )[0]
            coords_b = self._order_quads(self._decode_teacher_soft(heat))
            for j, idx in enumerate(batch_indices):
                name_j = self.image_list[idx]
                coords_j = coords_b[j].astype(np.float32, copy=False)
                preds[idx] = coords_j
                self.teacher_predictions[name_j] = coords_j
            batch_imgs = []
            batch_indices = []

        for idx, name in enumerate(self.image_list):
            if name.startswith("negative_"):
                continue

            img_path = self._locate_image_path(name, include_quarantine=True)
            if img_path is None or not img_path.exists():
                missing += 1
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                missing += 1
                continue

            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            batch_imgs.append(img)
            batch_indices.append(idx)

            if len(batch_imgs) >= batch_size:
                try:
                    flush_batch()
                except Exception as e:
                    print(f"Teacher batch inference failed (falling back to single-image): {e}")
                    # Fallback: run single-image for the current accumulated batch.
                    for ii in batch_indices:
                        nn = self.image_list[ii]
                        one = self._predict_teacher_single(nn)
                        if one is not None:
                            preds[ii] = np.asarray(one, dtype=np.float32)
                    batch_imgs = []
                    batch_indices = []
                    batch_size = 1

            processed += 1
            if processed % 500 == 0:
                print(f"  Teacher processed {processed} samples...")

        if batch_imgs:
            try:
                flush_batch()
            except Exception as e:
                print(f"Teacher final batch inference failed (falling back to single-image): {e}")
                for ii in batch_indices:
                    nn = self.image_list[ii]
                    one = self._predict_teacher_single(nn)
                    if one is not None:
                        preds[ii] = np.asarray(one, dtype=np.float32)

        self._save_teacher_predictions_cache(preds)
        if missing:
            print(f"Teacher precompute complete (missing/unreadable images: {missing}).")
        else:
            print("Teacher precompute complete.")

    def _ensure_teacher_predictions_for_overlay(self) -> None:
        """
        Ensure we have teacher predictions for *all* samples (used for the always-on purple overlay).

        This precomputes and caches bboxes for all non-negative samples the first time you run it.
        """
        if not self.teacher_onnx_path.exists():
            print(f"Teacher ONNX not found at {self.teacher_onnx_path}; skipping teacher overlay.")
            return
        if self._load_teacher_predictions_cache():
            return
        self._precompute_teacher_predictions(batch_size=min(int(self.batch_size or 32), 32))

    def _predict_teacher_single(self, name: str) -> Optional[np.ndarray]:
        if not self._load_teacher_session():
            return None

        img_size = int(self.teacher_img_size or 256)
        img_path = self._locate_image_path(name, include_quarantine=True)
        if img_path is None or not img_path.exists():
            self.status_message = f"Image not found: {name}"
            return None

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            self.status_message = f"Failed to load image: {name}"
            return None

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        batch = img[None, ...]  # [1,H,W,3] raw255
        x = self._teacher_convert_to_input(batch)

        heat = self.teacher_session.run(
            [self.teacher_output_name],
            {self.teacher_input_name: x},
        )[0]
        coords = self._order_quads(self._decode_teacher_soft(heat))[0]

        self.teacher_predictions[name] = coords
        return coords

    def get_teacher_prediction(self, name: str) -> Optional[np.ndarray]:
        if name in self.teacher_predictions:
            return self.teacher_predictions[name]
        if not self._teacher_cache_attempted:
            self._teacher_cache_attempted = True
            self._load_teacher_predictions_cache()
        if name in self.teacher_predictions:
            return self.teacher_predictions[name]
        return self._predict_teacher_single(name)

    def _ensure_quarantine_dirs(self) -> None:
        self.image_quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.label_quarantine_dir.mkdir(parents=True, exist_ok=True)

    def _primary_image_dir(self, name: str) -> Path:
        return self.dataset_path / ("images-negative" if name.startswith("negative_") else "images")

    def _locate_image_path(self, name: str, *, include_quarantine: bool = True) -> Optional[Path]:
        """Locate an image file for a given sample name."""
        search_dirs = [self._primary_image_dir(name)]
        if include_quarantine:
            search_dirs.append(self.image_quarantine_dir)

        stem = Path(name).stem
        for img_dir in search_dirs:
            for ext in IMAGE_EXTS:
                candidate = img_dir / f"{stem}{ext}"
                if candidate.exists():
                    return candidate
            candidate = img_dir / name
            if candidate.exists():
                return candidate
        return None

    def _locate_label_path(self, name: str) -> Optional[Path]:
        """Locate a label file for a given sample name."""
        stem = Path(name).stem
        for label_dir in [self.dataset_path / "labels", self.label_quarantine_dir]:
            candidate = label_dir / f"{stem}.txt"
            if candidate.exists():
                return candidate
        return None

    def _load_image_for_inference(self, name: str, img_size: int) -> np.ndarray:
        """Load and preprocess image for inference at a given input size."""
        img_path = self._locate_image_path(name, include_quarantine=True)
        if img_path is None or not img_path.exists():
            print(f"Warning: Image not found: {name}")
            return np.zeros((img_size, img_size, 3), dtype=np.float32)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Failed to load: {img_path}")
            return np.zeros((img_size, img_size, 3), dtype=np.float32)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))

        # ImageNet normalization
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img

    def _load_image(self, name: str) -> np.ndarray:
        """Load and preprocess image for inference."""
        return self._load_image_for_inference(name, self.img_size)

    def _load_image_display(self, name: str, display_size: int = 512) -> np.ndarray:
        """Load image for display (no normalization), at larger size for better viewing."""
        img_path = self._locate_image_path(name, include_quarantine=True)
        if img_path is None or not img_path.exists():
            return np.zeros((display_size, display_size, 3), dtype=np.uint8)

        img = cv2.imread(str(img_path))
        if img is None:
            return np.zeros((display_size, display_size, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (display_size, display_size))
        return img

    def _run_inference(self):
        """Run model inference on all images."""
        print(f"Running inference on {len(self.image_list)} images...")

        for i in range(0, len(self.image_list), self.batch_size):
            batch_names = self.image_list[i : i + self.batch_size]
            batch_images = np.stack([self._load_image(n) for n in batch_names])

            coords, score_logits = self.model.predict(batch_images, verbose=0)

            for j, name in enumerate(batch_names):
                self.predictions[name] = coords[j]
                # Safe sigmoid
                logit = float(score_logits[j].flatten()[0])
                self.scores[name] = 1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))

            if (i + self.batch_size) % 500 == 0 or i + self.batch_size >= len(self.image_list):
                print(f"  Processed {min(i + self.batch_size, len(self.image_list))}/{len(self.image_list)}")

        print("Inference complete.")

    def _compute_metrics(self):
        """Compute metrics for each image."""
        print("Computing metrics...")

        for name in self.image_list:
            gt = self.gt_labels.get(name, np.zeros(8))
            pred = self.predictions.get(name, np.zeros(8))

            # Skip negative samples for IoU/error computation
            is_negative = name.startswith("negative_") or np.allclose(gt, 0)

            if is_negative:
                self.metrics[name] = {
                    "iou": 0.0,
                    "corner_error_mean": 0.0,
                    "corner_error_max": 0.0,
                    "score": self.scores.get(name, 0.0),
                    "is_negative": True,
                    "gt_min_angle_deg": 180.0,
                }
            else:
                iou = compute_polygon_iou(pred, gt)
                mean_err, per_corner = compute_corner_error(pred, gt, self.img_size)
                gt_min_angle = self._compute_min_corner_angle_deg(gt)

                self.metrics[name] = {
                    "iou": iou,
                    "corner_error_mean": mean_err,
                    "corner_error_max": float(per_corner.max()),
                    "score": self.scores.get(name, 0.0),
                    "is_negative": False,
                    "gt_min_angle_deg": gt_min_angle,
                }

        print("Metrics computed.")

    def _compute_purple_metrics(self) -> None:
        """Compute IoU of the purple checkpoint vs current GT (per sample)."""
        if not self.purple_checkpoint_path.exists():
            return

        if not self.purple_predictions:
            return

        computed = 0
        for name in self.image_list:
            if self.metrics.get(name, {}).get("is_negative", False):
                self.metrics[name]["purple_iou"] = 0.0
                continue

            coords = self.purple_predictions.get(name)
            if coords is None:
                continue

            gt = self.modified.get(name, self.gt_labels.get(name, np.zeros(8)))
            self.metrics[name]["purple_iou"] = float(compute_polygon_iou(coords, gt))
            computed += 1

        if computed:
            print(f"Computed purple IoU for {computed} samples.")

    def _apply_filter_and_sort_keep_current(self, keep_name: Optional[str] = None) -> None:
        """Re-apply filter/sort and keep a given sample selected when possible."""
        if keep_name is None and self.filtered_list:
            keep_name = self.filtered_list[self.current_index]
        self._apply_filter_and_sort()
        if keep_name and keep_name in self.filtered_list:
            self.current_index = self.filtered_list.index(keep_name)

    def _apply_filter_and_sort(self):
        """Apply current filter and sort to image list."""
        # Filter
        if self.current_filter == "all":
            filtered = [
                n
                for n in self.image_list
                if not self.metrics[n]["is_negative"] and n not in self.discarded
            ]
        elif self.current_filter == "acute_gt":
            thresh = float(self.gt_acute_angle_threshold_deg)
            filtered = [
                n
                for n in self.image_list
                if not self.metrics[n]["is_negative"]
                and n not in self.discarded
                and float(self.metrics[n].get("gt_min_angle_deg", 180.0)) < thresh
            ]
        elif self.current_filter == "failed":
            filtered = [
                n
                for n in self.image_list
                if not self.metrics[n]["is_negative"]
                and n not in self.discarded
                and self.metrics[n]["iou"] < 0.5
            ]
        elif self.current_filter == "low_iou":
            filtered = [
                n
                for n in self.image_list
                if not self.metrics[n]["is_negative"]
                and n not in self.discarded
                and 0.5 <= self.metrics[n]["iou"] < 0.75
            ]
        elif self.current_filter == "medium_iou":
            filtered = [
                n
                for n in self.image_list
                if not self.metrics[n]["is_negative"]
                and n not in self.discarded
                and 0.75 <= self.metrics[n]["iou"] < 0.90
            ]
        elif self.current_filter == "high_iou":
            filtered = [
                n
                for n in self.image_list
                if not self.metrics[n]["is_negative"]
                and n not in self.discarded
                and 0.90 <= self.metrics[n]["iou"] < 0.95
            ]
        elif self.current_filter == "excellent":
            filtered = [
                n
                for n in self.image_list
                if not self.metrics[n]["is_negative"]
                and n not in self.discarded
                and self.metrics[n]["iou"] >= 0.95
            ]
        elif self.current_filter == "high_error":
            filtered = [
                n
                for n in self.image_list
                if not self.metrics[n]["is_negative"]
                and n not in self.discarded
                and self.metrics[n]["corner_error_mean"] > 10
            ]
        elif self.current_filter == "modified":
            filtered = [n for n in self.image_list if n in self.modified and n not in self.discarded]
        elif self.current_filter == "discarded":
            filtered = [n for n in self.image_list if n in self.discarded]
        elif self.current_filter == "negative":
            filtered = [n for n in self.image_list if self.metrics[n]["is_negative"] and n not in self.discarded]
        else:
            filtered = list(self.image_list)

        # Sort
        if self.current_sort == "error_desc":
            filtered.sort(key=lambda n: -self.metrics[n]["corner_error_mean"])
        elif self.current_sort == "error_asc":
            filtered.sort(key=lambda n: self.metrics[n]["corner_error_mean"])
        elif self.current_sort == "iou_asc":
            filtered.sort(key=lambda n: self.metrics[n]["iou"])
        elif self.current_sort == "iou_desc":
            filtered.sort(key=lambda n: -self.metrics[n]["iou"])
        elif self.current_sort == "gt_angle_asc":
            filtered.sort(key=lambda n: self.metrics[n].get("gt_min_angle_deg", 180.0))
        elif self.current_sort == "gt_angle_desc":
            filtered.sort(key=lambda n: -self.metrics[n].get("gt_min_angle_deg", 0.0))
        elif self.current_sort == "name":
            filtered.sort()

        self.filtered_list = filtered
        self.current_index = min(self.current_index, max(0, len(filtered) - 1))

    def get_dashboard_stats(self) -> str:
        """Get aggregate statistics for dashboard display."""
        positive_metrics = [
            m for n, m in self.metrics.items() if not m["is_negative"] and n not in self.discarded
        ]
        n_total = len(positive_metrics)

        if n_total == 0:
            return "No positive samples found."

        ious = [m["iou"] for m in positive_metrics]
        errors = [m["corner_error_mean"] for m in positive_metrics]
        purple_ious = [m.get("purple_iou") for m in positive_metrics if m.get("purple_iou") is not None]

        mean_iou = np.mean(ious)
        mean_error = np.mean(errors)
        recall_95 = sum(1 for iou in ious if iou >= 0.95) / n_total * 100
        recall_90 = sum(1 for iou in ious if iou >= 0.90) / n_total * 100
        n_failed = sum(1 for iou in ious if iou < 0.5)
        n_high_error = sum(1 for e in errors if e > 10)

        purple_block = ""
        if purple_ious:
            purple_n = len(purple_ious)
            purple_mean_iou = float(np.mean(purple_ious))
            purple_recall_95 = sum(1 for iou in purple_ious if float(iou) >= 0.95) / purple_n * 100
            purple_recall_90 = sum(1 for iou in purple_ious if float(iou) >= 0.90) / purple_n * 100
            purple_failed = sum(1 for iou in purple_ious if float(iou) < 0.5)
            purple_block = (
                f"| **Mean IoU (Purple)** | {purple_mean_iou:.4f} |\n"
                f"| **Δ Mean IoU (Purple - Pred)** | {purple_mean_iou - float(mean_iou):+.4f} |\n"
                f"| **Recall@95 (Purple)** | {purple_recall_95:.1f}% |\n"
                f"| **Recall@90 (Purple)** | {purple_recall_90:.1f}% |\n"
                f"| **Failed (Purple IoU<0.5)** | {purple_failed} |\n"
            )

        stats = f"""### Model Quality Dashboard

| Metric | Value |
|--------|-------|
| **Mean IoU (Pred)** | {mean_iou:.4f} |
| **Mean Error (Pred)** | {mean_error:.1f}px |
| **Recall@95 (Pred)** | {recall_95:.1f}% |
| **Recall@90 (Pred)** | {recall_90:.1f}% |
| **Failed (Pred IoU<0.5)** | {n_failed} |
| **High Error (Pred >10px)** | {n_high_error} |
{purple_block}\
| **Total Positive** | {n_total} |
| **Modified** | {len(self.modified)} |
| **Discarded** | {len(self.discarded)} |
"""
        return stats

    def get_current_image_info(self) -> str:
        """Get info for current image."""
        if not self.filtered_list:
            if self.current_filter == "acute_gt":
                return (
                    f"No images match filter: GT min angle < {self.gt_acute_angle_threshold_deg:.0f}°.\n\n"
                    "Increase the threshold slider."
                )
            return "No images match filter."

        name = self.filtered_list[self.current_index]
        m = self.metrics[name]

        tags = []
        if name in self.discarded:
            tags.append("DISCARDED")
        if name in self.proposals:
            tags.append("PROPOSAL")
        if name in self.modified:
            tags.append("MODIFIED")
        status = " ".join(f"**[{t}]**" for t in tags)

        gt_angle = m.get("gt_min_angle_deg", 0.0)
        gt_angle_str = "-" if m.get("is_negative", False) else f"{gt_angle:.1f}°"

        purple_iou_str = "-"
        purple_delta_str = "-"
        if not m.get("is_negative", False):
            purple_iou = m.get("purple_iou")
            if purple_iou is None:
                coords = self.purple_predictions.get(name)
                if coords is None:
                    coords = self.get_purple_prediction(name)
                if coords is not None:
                    purple_iou = float(compute_polygon_iou(coords, self.get_current_corners()))
                    self.metrics[name]["purple_iou"] = purple_iou
            if purple_iou is not None:
                purple_iou_str = f"{float(purple_iou):.4f}"
                try:
                    purple_delta_str = f"{float(purple_iou) - float(m.get('iou', 0.0)):+.4f}"
                except Exception:
                    purple_delta_str = "-"

        info = f"""### {name} {status}

| Metric | Value |
|--------|-------|
| **IoU (Pred vs GT)** | {m['iou']:.4f} |
| **IoU (Purple vs GT)** | {purple_iou_str} |
| **Δ IoU (Purple - Pred)** | {purple_delta_str} |
| **Error (mean)** | {m['corner_error_mean']:.1f}px |
| **Error (max)** | {m['corner_error_max']:.1f}px |
| **Score** | {m['score']:.3f} |
| **GT min angle** | {gt_angle_str} |

Position: {self.current_index + 1} / {len(self.filtered_list)}
"""
        if name in self.proposals and not m["is_negative"] and name not in self.discarded:
            proposal = self.proposals[name]
            pred = self.predictions.get(name, np.zeros(8))
            p_iou = compute_polygon_iou(pred, proposal)
            p_mean, p_per_corner = compute_corner_error(pred, proposal, self.img_size)
            info += (
                f"\n**Proposal vs pred**: IoU {p_iou:.4f}, Error(mean) {p_mean:.1f}px, "
                f"Error(max) {float(p_per_corner.max()):.1f}px\n"
            )
        return info

    def get_current_corners(self) -> np.ndarray:
        """Get current GT corners (modified if available)."""
        if not self.filtered_list:
            return np.zeros(8)

        name = self.filtered_list[self.current_index]
        if name in self.modified:
            return self.modified[name]
        return self.gt_labels.get(name, np.zeros(8))

    def get_current_editable_corners(self) -> np.ndarray:
        """Get corners currently being edited (proposal if present, else GT)."""
        if not self.filtered_list:
            return np.zeros(8)
        name = self.filtered_list[self.current_index]
        if name in self.proposals:
            return self.proposals[name]
        return self.get_current_corners()

    def save_proposal_changes(self, corners: np.ndarray):
        """Save proposal corners (pending approval)."""
        if not self.filtered_list:
            return
        name = self.filtered_list[self.current_index]
        corners = np.clip(np.array(corners, dtype=np.float32), 0.0, 1.0)
        self.proposals[name] = corners
        self.status_message = f"Updated proposal for {name} (not applied)"

    def propose_from_candidate_checkpoint(self):
        """Create a proposal bbox from the candidate checkpoint prediction."""
        if not self.filtered_list:
            return
        name = self.filtered_list[self.current_index]
        coords = self.get_candidate_prediction(name)
        if coords is None:
            if not self.status_message:
                self.status_message = "Failed to load candidate prediction"
            return
        self.proposals[name] = np.array(coords, dtype=np.float32).copy()
        ckpt = self.candidate_checkpoint_path.name
        self.status_message = f"Proposed bbox from {ckpt} for {name} (Approve/Cancel to apply)"

    def propose_from_teacher_onnx(self):
        """Create a proposal bbox from the ONNX teacher heatmap model."""
        if not self.filtered_list:
            return
        name = self.filtered_list[self.current_index]
        coords = self.get_teacher_prediction(name)
        if coords is None:
            if not self.status_message:
                self.status_message = "Failed to load teacher prediction"
            return
        self.proposals[name] = np.array(coords, dtype=np.float32).copy()
        model = self.teacher_onnx_path.name
        self.status_message = f"Proposed bbox from {model} for {name} (Approve/Cancel to apply)"

    def approve_current_proposal(self):
        """Approve proposal for current image and apply as GT."""
        if not self.filtered_list:
            return
        name = self.filtered_list[self.current_index]
        if name not in self.proposals:
            self.status_message = "No proposal to approve"
            return
        proposal = self.proposals.pop(name)
        self.save_gt_changes(proposal)

    def cancel_current_proposal(self):
        """Discard proposal for current image."""
        if not self.filtered_list:
            return
        name = self.filtered_list[self.current_index]
        if name in self.proposals:
            self.proposals.pop(name, None)
            self.status_message = f"Discarded proposal for {name}"

    def get_current_overlay_state_json(self) -> str:
        """JSON payload consumed by the frontend overlay (gt/pred + sample info)."""
        if not self.filtered_list:
            return json.dumps({"name": None})

        name = self.filtered_list[self.current_index]
        gt = self.get_current_corners().astype(float).tolist()
        pred = self.predictions.get(name, np.zeros(8)).astype(float).tolist()
        proposal = self.proposals.get(name)
        edit_target = "proposal" if proposal is not None else "gt"
        purple = None
        if not name.startswith("negative_"):
            coords = self.purple_predictions.get(name)
            if coords is None:
                coords = self.get_purple_prediction(name)
            if coords is not None:
                coords = np.asarray(coords, dtype=np.float32)
                if float(np.sum(np.abs(coords))) > 1e-6:
                    purple = coords.astype(float).tolist()
        return json.dumps(
            {
                "name": name,
                "gt": gt,
                "pred": pred,
                "proposal": proposal.astype(float).tolist() if proposal is not None else None,
                "purple": purple,
                "edit_target": edit_target,
                "display_size": int(self.display_size),
                "discarded": name in self.discarded,
                "modified": name in self.modified,
            }
        )

    def render_image(
        self,
        editing_corners: Optional[np.ndarray] = None,
        show_handles: bool = True,
        draw_overlays: bool = True,
    ) -> np.ndarray:
        """Render image (optionally with GT and prediction overlays)."""
        if not self.filtered_list:
            return np.zeros((self.display_size, self.display_size, 3), dtype=np.uint8)

        name = self.filtered_list[self.current_index]
        img = self._load_image_display(name, self.display_size)

        if not draw_overlays:
            return img

        # Get coords
        gt_coords = self.get_current_corners()
        pred_coords = self.predictions.get(name, np.zeros(8))

        if editing_corners is not None:
            gt_coords = editing_corners

        # Draw prediction (red, dashed)
        self._draw_polygon(img, pred_coords, color=(255, 100, 100), label="Pred", dashed=True)

        # Draw GT (green, solid) with larger draggable handles
        self._draw_polygon(img, gt_coords, color=(100, 255, 100), label="GT", dashed=False,
                          draw_handles=show_handles, selected_corner=self.selected_corner)

        return img

    def handle_image_click(self, coords: np.ndarray, click_x: float, click_y: float) -> Tuple[np.ndarray, bool]:
        """
        Handle click on image. Returns (new_coords, changed).

        If no corner selected: select nearest corner if within radius
        If corner selected: move that corner to click position
        """
        h = w = self.img_size

        # Convert click to pixel coords (Gradio gives normalized 0-1 for some image types)
        px = int(click_x)
        py = int(click_y)

        if self.selected_corner is None:
            # Try to select a corner
            for i in range(4):
                cx = int(coords[i * 2] * w)
                cy = int(coords[i * 2 + 1] * h)
                dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                if dist <= CORNER_CLICK_RADIUS:
                    self.selected_corner = i
                    return coords, False  # No coord change, just selection
            return coords, False
        else:
            # Move selected corner to click position
            new_coords = coords.copy()
            new_coords[self.selected_corner * 2] = np.clip(px / w, 0, 1)
            new_coords[self.selected_corner * 2 + 1] = np.clip(py / h, 0, 1)
            self.selected_corner = None  # Deselect after move
            return new_coords, True

    def _draw_polygon(
        self,
        img: np.ndarray,
        coords: np.ndarray,
        color: Tuple[int, int, int],
        label: str,
        dashed: bool = False,
        draw_handles: bool = False,
        selected_corner: Optional[int] = None,
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

            if dashed:
                # Dashed line
                self._draw_dashed_line(img, p1, p2, color, thickness=2)
            else:
                cv2.line(img, p1, p2, color, thickness=2)

        # Draw corner circles and labels
        for i, (x, y) in enumerate(points):
            if draw_handles:
                # Large draggable handles for GT
                radius = 14 if selected_corner == i else 10
                # Outer ring (highlight if selected)
                if selected_corner == i:
                    cv2.circle(img, (x, y), radius + 4, (255, 255, 0), 3)  # Yellow highlight
                cv2.circle(img, (x, y), radius, color, -1)
                cv2.circle(img, (x, y), radius, (255, 255, 255), 2)
                # Inner dot
                cv2.circle(img, (x, y), 3, (255, 255, 255), -1)
            else:
                # Small markers for predictions
                cv2.circle(img, (x, y), 5, color, -1)
                cv2.circle(img, (x, y), 5, (0, 0, 0), 1)

            # Corner label
            label_pos = (x + 12, y - 12)
            cv2.putText(img, CORNER_LABELS[i], label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(img, CORNER_LABELS[i], label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _draw_dashed_line(
        self,
        img: np.ndarray,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        dash_length: int = 8,
    ):
        """Draw dashed line."""
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx * dx + dy * dy)

        if dist < 1:
            return

        n_dashes = int(dist / dash_length)
        if n_dashes < 1:
            n_dashes = 1

        for i in range(0, n_dashes, 2):
            t1 = i / n_dashes
            t2 = min((i + 1) / n_dashes, 1.0)
            start = (int(x1 + dx * t1), int(y1 + dy * t1))
            end = (int(x1 + dx * t2), int(y1 + dy * t2))
            cv2.line(img, start, end, color, thickness)

    @staticmethod
    def _compute_min_corner_angle_deg(coords: np.ndarray) -> float:
        """Compute minimum corner angle (degrees) for a quad (TL,TR,BR,BL)."""
        pts = np.array(coords, dtype=np.float32).reshape(4, 2)
        angles = []
        eps = 1e-9
        for i in range(4):
            p_prev = pts[(i - 1) % 4]
            p = pts[i]
            p_next = pts[(i + 1) % 4]
            v1 = p_prev - p
            v2 = p_next - p
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < eps or n2 < eps:
                angles.append(0.0)
                continue
            cos = float(np.dot(v1, v2) / (n1 * n2))
            cos = float(np.clip(cos, -1.0, 1.0))
            angles.append(float(np.degrees(np.arccos(cos))))
        return float(np.min(angles)) if angles else 0.0

    # === Actions ===

    def use_prediction_as_gt(self):
        """Copy prediction to GT (modified)."""
        if not self.filtered_list:
            return

        name = self.filtered_list[self.current_index]
        self.save_gt_changes(self.predictions[name].copy())

    def save_gt_changes(self, corners: np.ndarray):
        """Save modified GT corners."""
        if not self.filtered_list:
            return

        name = self.filtered_list[self.current_index]
        corners = np.clip(np.array(corners, dtype=np.float32), 0.0, 1.0)
        self.modified[name] = corners
        self.status_message = f"Updated GT for {name} (not saved to disk yet)"

        # Update metrics
        pred = self.predictions.get(name, np.zeros(8))
        iou = compute_polygon_iou(pred, corners)
        mean_err, per_corner = compute_corner_error(pred, corners, self.img_size)
        gt_min_angle = self._compute_min_corner_angle_deg(corners)

        self.metrics[name]["iou"] = iou
        self.metrics[name]["corner_error_mean"] = mean_err
        self.metrics[name]["corner_error_max"] = float(per_corner.max())
        self.metrics[name]["gt_min_angle_deg"] = gt_min_angle
        purple = self.purple_predictions.get(name)
        if purple is not None:
            self.metrics[name]["purple_iou"] = float(compute_polygon_iou(purple, corners))
        self._apply_filter_and_sort_keep_current(keep_name=name)

    def discard_current(self):
        """Discard current image (move image/label to quarantine)."""
        if not self.filtered_list:
            return

        name = self.filtered_list[self.current_index]
        if name in self.discarded:
            return

        self._ensure_quarantine_dirs()

        # Move image to quarantine (if still in the primary directory)
        img_src = self._locate_image_path(name, include_quarantine=False)
        if img_src is not None and img_src.exists():
            img_dst = self.image_quarantine_dir / img_src.name
            try:
                shutil.move(str(img_src), str(img_dst))
            except Exception as e:
                print(f"Warning: failed to quarantine image {img_src} -> {img_dst}: {e}")

        # Move label to quarantine (positive samples only)
        if not name.startswith("negative_"):
            stem = Path(name).stem
            label_src = self.dataset_path / "labels" / f"{stem}.txt"
            if label_src.exists():
                label_dst = self.label_quarantine_dir / label_src.name
                try:
                    shutil.move(str(label_src), str(label_dst))
                except Exception as e:
                    print(f"Warning: failed to quarantine label {label_src} -> {label_dst}: {e}")

        self.discarded.add(name)
        self.modified.pop(name, None)
        self.proposals.pop(name, None)
        self.status_message = f"Quarantined {name} -> {IMAGE_QUARANTINE_DIRNAME}/, {LABEL_QUARANTINE_DIRNAME}/"
        self._save_discarded_list()
        self._apply_filter_and_sort()

    def undiscard_current(self):
        """Restore current image from quarantine."""
        if not self.filtered_list:
            return

        name = self.filtered_list[self.current_index]
        if name not in self.discarded:
            return

        # Move image back (if present in quarantine)
        img_src = self._locate_image_path(name, include_quarantine=True)
        if img_src is not None and img_src.exists() and img_src.parent == self.image_quarantine_dir:
            img_dst_dir = self._primary_image_dir(name)
            img_dst_dir.mkdir(parents=True, exist_ok=True)
            img_dst = img_dst_dir / img_src.name
            try:
                shutil.move(str(img_src), str(img_dst))
            except Exception as e:
                print(f"Warning: failed to restore image {img_src} -> {img_dst}: {e}")

        # Move label back (if present in quarantine)
        if not name.startswith("negative_"):
            stem = Path(name).stem
            label_src = self.label_quarantine_dir / f"{stem}.txt"
            if label_src.exists():
                label_dst_dir = self.dataset_path / "labels"
                label_dst_dir.mkdir(parents=True, exist_ok=True)
                label_dst = label_dst_dir / label_src.name
                try:
                    shutil.move(str(label_src), str(label_dst))
                except Exception as e:
                    print(f"Warning: failed to restore label {label_src} -> {label_dst}: {e}")

        self.discarded.discard(name)
        self.status_message = f"Restored {name} (from quarantine)"
        self._save_discarded_list()
        self._apply_filter_and_sort()

    def undo_modifications(self):
        """Undo modifications to current image."""
        if not self.filtered_list:
            return

        name = self.filtered_list[self.current_index]
        if name in self.modified:
            del self.modified[name]
            self.status_message = f"Reverted GT for {name}"

            # Recompute metrics with original GT
            gt = self.gt_labels.get(name, np.zeros(8))
            pred = self.predictions.get(name, np.zeros(8))
            iou = compute_polygon_iou(pred, gt)
            mean_err, per_corner = compute_corner_error(pred, gt, self.img_size)
            gt_min_angle = self._compute_min_corner_angle_deg(gt)

            self.metrics[name]["iou"] = iou
            self.metrics[name]["corner_error_mean"] = mean_err
            self.metrics[name]["corner_error_max"] = float(per_corner.max())
            self.metrics[name]["gt_min_angle_deg"] = gt_min_angle
            purple = self.purple_predictions.get(name)
            if purple is not None:
                self.metrics[name]["purple_iou"] = float(compute_polygon_iou(purple, gt))
            self._apply_filter_and_sort_keep_current(keep_name=name)

    def reset_selection(self):
        """Reset corner selection."""
        self.selected_corner = None

    def _save_discarded_list(self) -> None:
        discarded_path = self.dataset_path / DISCARDED_FILENAME
        with open(discarded_path, "w") as f:
            for name in sorted(self.discarded):
                f.write(name + "\n")

    def save_all_to_disk(self):
        """Save all modifications and discarded list to disk."""
        saved_count = 0
        skipped_discarded = 0

        # Save modified labels
        for name, coords in self.modified.items():
            if name in self.discarded:
                skipped_discarded += 1
                continue
            stem = Path(name).stem
            label_path = self.dataset_path / "labels" / f"{stem}.txt"

            # Format: class_id x0 y0 x1 y1 x2 y2 x3 y3
            line = "0 " + " ".join(f"{c:.6f}" for c in coords)

            with open(label_path, "w") as f:
                f.write(line + "\n")

            saved_count += 1

        self._save_discarded_list()

        extra = f" (skipped {skipped_discarded} discarded)" if skipped_discarded else ""
        self.status_message = (
            f"Saved {saved_count} labels to labels/*.txt{extra}; "
            f"{len(self.discarded)} discarded in {DISCARDED_FILENAME}"
        )
        return self.status_message

    def export_clean_split(self) -> str:
        """Export new split file excluding discarded images."""
        # Load original split
        split_files = [
            self.dataset_path / f"{self.split}_with_negative_v2.txt",
            self.dataset_path / f"{self.split}_with_negative.txt",
            self.dataset_path / f"{self.split}.txt",
        ]

        original_path = None
        for sf in split_files:
            if sf.exists():
                original_path = sf
                break

        if original_path is None:
            self.status_message = "Error: Original split file not found"
            return self.status_message

        # Read original (keeping original format)
        with open(original_path) as f:
            original_lines = f.readlines()

        # Filter out discarded
        clean_lines = []
        for line in original_lines:
            name = line.strip()
            if not name:
                continue

            # Normalize for comparison
            check_name = name
            if check_name.startswith("images-negative/"):
                check_name = check_name[len("images-negative/"):]
            elif check_name.startswith("images/"):
                check_name = check_name[len("images/"):]

            if check_name not in self.discarded:
                clean_lines.append(line)

        # Write clean split
        clean_path = self.dataset_path / f"{self.split}_cleaned.txt"
        with open(clean_path, "w") as f:
            f.writelines(clean_lines)

        self.status_message = (
            f"Exported {len(clean_lines)} images to {clean_path.name} (removed {len(self.discarded)})"
        )
        return self.status_message


def create_app(app: OutlierReviewApp) -> gr.Blocks:
    """Create Gradio interface."""

    def refresh_all():
        corners = app.get_current_editable_corners()
        img = app.render_image(draw_overlays=False)
        overlay_state = app.get_current_overlay_state_json()
        info = app.get_current_image_info()
        stats = app.get_dashboard_stats()

        pos = f"{app.current_index + 1}" if app.filtered_list else "0"
        total = f"/ {len(app.filtered_list)}"
        angle_slider_update = gr.update(
            value=float(app.gt_acute_angle_threshold_deg),
            visible=(app.current_filter == "acute_gt"),
        )

        return (
            img,
            overlay_state,
            info,
            stats,
            float(corners[0]),
            float(corners[1]),  # TL
            float(corners[2]),
            float(corners[3]),  # TR
            float(corners[4]),
            float(corners[5]),  # BR
            float(corners[6]),
            float(corners[7]),  # BL
            pos,
            total,
            angle_slider_update,
            app.status_message,
        )

    def update_filter(filter_choice):
        filter_map = {
            "All Positive": "all",
            "GT Acute Angles": "acute_gt",
            "Failed (IoU < 0.5)": "failed",
            "Low IoU (0.5-0.75)": "low_iou",
            "Medium IoU (0.75-0.90)": "medium_iou",
            "High IoU (0.90-0.95)": "high_iou",
            "Excellent (IoU > 0.95)": "excellent",
            "High Error (>10px)": "high_error",
            "Modified": "modified",
            "Discarded": "discarded",
            "Negative Samples": "negative",
        }
        app.current_filter = filter_map.get(filter_choice, "all")
        app.current_index = 0
        app.status_message = ""
        app._apply_filter_and_sort()
        return refresh_all()

    def update_acute_angle_threshold(threshold_deg: float):
        try:
            app.gt_acute_angle_threshold_deg = float(threshold_deg)
        except Exception:
            return refresh_all()
        keep_name = app.filtered_list[app.current_index] if app.filtered_list else None
        app._apply_filter_and_sort_keep_current(keep_name=keep_name)
        return refresh_all()

    def update_sort(sort_choice):
        sort_map = {
            "Error (high to low)": "error_desc",
            "Error (low to high)": "error_asc",
            "IoU (low to high)": "iou_asc",
            "IoU (high to low)": "iou_desc",
            "GT min angle (low to high)": "gt_angle_asc",
            "GT min angle (high to low)": "gt_angle_desc",
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
        if app.filtered_list and app.filtered_list[app.current_index] in app.proposals:
            app.save_proposal_changes(corners)
        else:
            app.save_gt_changes(corners)
        return refresh_all()

    def apply_drag(edit_state_json: str):
        try:
            payload = json.loads(edit_state_json)
        except Exception:
            return refresh_all()

        if not app.filtered_list:
            return refresh_all()

        name = payload.get("name")
        target = payload.get("target", "gt")
        corners = payload.get("coords") or payload.get("gt")
        if name != app.filtered_list[app.current_index]:
            return refresh_all()
        if not isinstance(corners, list) or len(corners) != 8:
            return refresh_all()

        if target == "proposal":
            app.save_proposal_changes(np.array(corners, dtype=np.float32))
        else:
            app.save_gt_changes(np.array(corners, dtype=np.float32))
        return refresh_all()

    def use_prediction():
        app.use_prediction_as_gt()
        return refresh_all()

    def propose_256():
        app.propose_from_candidate_checkpoint()
        return refresh_all()

    def propose_teacher():
        app.propose_from_teacher_onnx()
        return refresh_all()

    def approve_proposal():
        app.approve_current_proposal()
        return refresh_all()

    def cancel_proposal():
        app.cancel_current_proposal()
        return refresh_all()

    def discard_image():
        app.discard_current()
        return refresh_all()

    def undiscard_image():
        app.undiscard_current()
        return refresh_all()

    def undo_changes():
        app.undo_modifications()
        app.reset_selection()
        return refresh_all()

    def save_all():
        result = app.save_all_to_disk()
        stats = app.get_dashboard_stats()
        return stats, result

    def export_split():
        result = app.export_clean_split()
        return result

    # Build interface
    with gr.Blocks(title="DocCornerNet Outlier Review") as demo:
        gr.Markdown("# DocCornerNet Outlier Review")

        with gr.Group(elem_id="action-bar"):
            with gr.Row(variant="compact"):
                filter_dropdown = gr.Dropdown(
                    choices=[
                        "All Positive",
                        "GT Acute Angles",
                        "Failed (IoU < 0.5)",
                        "Low IoU (0.5-0.75)",
                        "Medium IoU (0.75-0.90)",
                        "High IoU (0.90-0.95)",
                        "Excellent (IoU > 0.95)",
                        "High Error (>10px)",
                        "Modified",
                        "Discarded",
                        "Negative Samples",
                    ],
                    value="All Positive",
                    label="Filter",
                )
                sort_dropdown = gr.Dropdown(
                    choices=[
                        "Error (high to low)",
                        "Error (low to high)",
                        "IoU (low to high)",
                        "IoU (high to low)",
                        "GT min angle (low to high)",
                        "GT min angle (high to low)",
                        "Filename",
                    ],
                    value="Error (high to low)",
                    label="Sort",
                )

                prev_btn = gr.Button("◀ Prev")
                next_btn = gr.Button("Next ▶")
                position_input = gr.Textbox(value="1", label="Position", max_lines=1)
                total_label = gr.Markdown(f"/ {len(app.filtered_list)}")

            with gr.Row(variant="compact"):
                acute_angle_slider = gr.Slider(
                    minimum=5,
                    maximum=90,
                    step=1,
                    value=float(app.gt_acute_angle_threshold_deg),
                    label="GT min angle < (deg)",
                    interactive=True,
                    visible=False,
                )

            with gr.Row(variant="compact"):
                propose_256_btn = gr.Button("📌 Propose (256)", variant="secondary")
                propose_teacher_btn = gr.Button("📌 Propose (SA24 ONNX)", variant="secondary")
                approve_proposal_btn = gr.Button("✅ Approve", variant="primary")
                cancel_proposal_btn = gr.Button("✖ Cancel", variant="secondary")
                use_pred_btn = gr.Button("🎯 Use Pred as GT", variant="secondary")
                undo_btn = gr.Button("↩️ Undo", variant="secondary")
                discard_btn = gr.Button("🗑️ Discard → Quarantine", variant="stop")
                undiscard_btn = gr.Button("♻️ Restore", variant="secondary")
                save_all_btn = gr.Button("💾 Save All to Disk", variant="primary")
                export_btn = gr.Button("📤 Export Clean Split")

            status_msg = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            with gr.Column(scale=2, min_width=520):
                image_display = gr.Image(
                    show_label=False,
                    interactive=False,
                    height=app.display_size,
                    width=app.display_size,
                    elem_id="review-image",
                )

                # Hidden fields used by custom JS overlay
                overlay_state_box = gr.Textbox(
                    value="",
                    elem_id="overlay-state",
                    interactive=False,
                    show_label=False,
                    container=False,
                    elem_classes="doccorner-hidden",
                )
                edit_state_box = gr.Textbox(
                    value="",
                    elem_id="edit-state",
                    interactive=True,
                    show_label=False,
                    container=False,
                    elem_classes="doccorner-hidden",
                )
                apply_drag_btn = gr.Button(
                    "apply_drag",
                    elem_id="apply-drag",
                    elem_classes="doccorner-hidden",
                    size="sm",
                )

            with gr.Column(scale=1, min_width=360):
                info_display = gr.Markdown(app.get_current_image_info())

                with gr.Accordion("📊 Model Quality Dashboard", open=False):
                    stats_display = gr.Markdown(app.get_dashboard_stats())

                with gr.Accordion("✏️ Fine-tune corners (auto-applies on release)", open=False):
                    with gr.Row():
                        tl_x = gr.Slider(0, 1, step=0.001, label="TL X", value=0, interactive=True)
                        tl_y = gr.Slider(0, 1, step=0.001, label="TL Y", value=0, interactive=True)
                    with gr.Row():
                        tr_x = gr.Slider(0, 1, step=0.001, label="TR X", value=1, interactive=True)
                        tr_y = gr.Slider(0, 1, step=0.001, label="TR Y", value=0, interactive=True)
                    with gr.Row():
                        br_x = gr.Slider(0, 1, step=0.001, label="BR X", value=1, interactive=True)
                        br_y = gr.Slider(0, 1, step=0.001, label="BR Y", value=1, interactive=True)
                    with gr.Row():
                        bl_x = gr.Slider(0, 1, step=0.001, label="BL X", value=0, interactive=True)
                        bl_y = gr.Slider(0, 1, step=0.001, label="BL Y", value=1, interactive=True)

        all_sliders = [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y]
        all_outputs = (
            [image_display, overlay_state_box, info_display, stats_display]
            + all_sliders
            + [position_input, total_label, acute_angle_slider, status_msg]
        )

        # Event handlers
        filter_dropdown.change(update_filter, filter_dropdown, all_outputs)
        sort_dropdown.change(update_sort, sort_dropdown, all_outputs)

        prev_btn.click(go_prev, None, all_outputs)
        next_btn.click(go_next, None, all_outputs)
        position_input.submit(go_to_index, position_input, all_outputs)
        acute_angle_slider.release(update_acute_angle_threshold, acute_angle_slider, all_outputs)

        for slider in all_sliders:
            slider.release(apply_corners, all_sliders, all_outputs)

        apply_drag_btn.click(
            apply_drag,
            edit_state_box,
            all_outputs,
            js="(v) => { const p = window.__doccorner_drag_payload || v; window.__doccorner_drag_payload = null; return p; }",
        )

        propose_256_btn.click(propose_256, None, all_outputs)
        propose_teacher_btn.click(propose_teacher, None, all_outputs)
        approve_proposal_btn.click(approve_proposal, None, all_outputs)
        cancel_proposal_btn.click(cancel_proposal, None, all_outputs)
        use_pred_btn.click(use_prediction, None, all_outputs)
        discard_btn.click(discard_image, None, all_outputs)
        undiscard_btn.click(undiscard_image, None, all_outputs)
        undo_btn.click(undo_changes, None, all_outputs)

        save_all_btn.click(save_all, None, [stats_display, status_msg])
        export_btn.click(export_split, None, status_msg)

        demo.load(refresh_all, None, all_outputs)

    return demo


CUSTOM_CSS = """
#action-bar {
  position: sticky;
  top: 0;
  z-index: 1000;
  background: var(--background-fill-primary);
  border-bottom: 1px solid var(--border-color-primary);
  padding: 8px;
}

.doccorner-hidden {
  position: absolute !important;
  left: -10000px !important;
  top: -10000px !important;
  width: 1px !important;
  height: 1px !important;
  opacity: 0 !important;
  overflow: hidden !important;
}

.doccorner-overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  z-index: 10;
  pointer-events: auto;
  cursor: crosshair;
}

#review-image img {
  user-select: none;
}
"""

CUSTOM_JS = r"""
(() => {
  const IMAGE_CONTAINER_ID = "review-image";
  const OVERLAY_STATE_ID = "overlay-state";
  const APPLY_DRAG_BUTTON_ID = "apply-drag";
  const CORNER_LABELS = ["TL", "TR", "BR", "BL"];

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  function findInputEl(root) {
    if (!root) return null;
    return root.querySelector("textarea, input") || root;
  }

  function setup() {
    const container = document.getElementById(IMAGE_CONTAINER_ID);
    const stateRoot = document.getElementById(OVERLAY_STATE_ID);
    const applyBtnRoot = document.getElementById(APPLY_DRAG_BUTTON_ID);
    if (!container || !stateRoot || !applyBtnRoot) return false;

    const stateEl = findInputEl(stateRoot);
    if (!stateEl) return false;

    const applyBtn = applyBtnRoot.tagName === "BUTTON" ? applyBtnRoot : applyBtnRoot.querySelector("button");
    if (!applyBtn) return false;

    let img = container.querySelector("img");
    if (!img) return false;

    let host = img.parentElement;
    if (!host) return false;
    host.style.position = "relative";

    let canvas = host.querySelector("canvas.doccorner-overlay");
    if (!canvas) {
      canvas = document.createElement("canvas");
      canvas.className = "doccorner-overlay";
      host.appendChild(canvas);
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return false;

    let current = null;
    let lastStateStr = null;
    let dragging = false;
    let dragCorner = -1;

    function resizeCanvas() {
      const rect = img.getBoundingClientRect();
      if (!rect.width || !rect.height) return null;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { w: rect.width, h: rect.height };
    }

    function drawPoly(coords, color, dashed) {
      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      const pts = [];
      for (let i = 0; i < 4; i++) {
        pts.push([coords[i * 2] * w, coords[i * 2 + 1] * h]);
      }
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash(dashed ? [8, 6] : []);
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < 4; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.closePath();
      ctx.stroke();
      ctx.restore();
      return pts;
    }

    function draw() {
      const size = resizeCanvas();
      if (!size) return;
      ctx.clearRect(0, 0, size.w, size.h);
      if (!current || !current.gt || !current.pred) return;
      if (!Array.isArray(current.gt) || current.gt.length !== 8) return;
      if (!Array.isArray(current.pred) || current.pred.length !== 8) return;

      drawPoly(current.pred, "rgba(255,80,80,0.95)", true);
      if (Array.isArray(current.purple) && current.purple.length === 8) {
        const s = current.purple.reduce((acc, v) => acc + Math.abs(v), 0);
        if (s > 1e-6) drawPoly(current.purple, "rgba(190,0,255,0.90)", false);
      }
      const hasProposal = Array.isArray(current.proposal) && current.proposal.length === 8;
      const editTarget = current.edit_target || (hasProposal ? "proposal" : "gt");

      const gtPts = drawPoly(current.gt, "rgba(80,255,80,0.90)", false);
      const proposalPts = hasProposal
        ? drawPoly(current.proposal, "rgba(80,160,255,0.95)", false)
        : null;

      const editingProposal = hasProposal && editTarget === "proposal";
      const activePts = editingProposal && proposalPts ? proposalPts : gtPts;
      const handleFill = editingProposal ? "rgba(80,160,255,0.30)" : "rgba(80,255,80,0.30)";

      for (let i = 0; i < 4; i++) {
        const [x, y] = activePts[i];
        const isActive = dragging && dragCorner === i;
        const r = isActive ? 12 : 9;

        if (isActive) {
          ctx.beginPath();
          ctx.arc(x, y, r + 5, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(255,255,0,0.95)";
          ctx.lineWidth = 3;
          ctx.stroke();
        }

        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        // Semi-transparent fill so you can see what's under the handle
        ctx.fillStyle = handleFill;
        ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.95)";
        ctx.lineWidth = 2;
        ctx.stroke();

        // Center dot (to see exact corner)
        ctx.beginPath();
        ctx.arc(x, y, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255,255,255,0.95)";
        ctx.fill();
        ctx.strokeStyle = "rgba(0,0,0,0.75)";
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = "rgba(0,0,0,0.9)";
        ctx.font = "12px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial";
        ctx.fillText(CORNER_LABELS[i], x + 12, y - 10);
      }
    }

    function eventPos(e) {
      const rect = canvas.getBoundingClientRect();
      return { x: e.clientX - rect.left, y: e.clientY - rect.top, w: rect.width, h: rect.height };
    }

    function pickCorner(px, py) {
      if (!current) return -1;
      const hasProposal = Array.isArray(current.proposal) && current.proposal.length === 8;
      const editTarget = current.edit_target || (hasProposal ? "proposal" : "gt");
      const coords =
        hasProposal && editTarget === "proposal" ? current.proposal : current.gt;
      if (!Array.isArray(coords) || coords.length !== 8) return -1;
      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      let best = -1;
      let bestDist = Infinity;
      for (let i = 0; i < 4; i++) {
        const cx = coords[i * 2] * w;
        const cy = coords[i * 2 + 1] * h;
        const d = Math.hypot(px - cx, py - cy);
        if (d < bestDist) {
          bestDist = d;
          best = i;
        }
      }
      return bestDist <= 18 ? best : -1;
    }

    function sendUpdate() {
      if (!current || !current.name) return;
      const hasProposal = Array.isArray(current.proposal) && current.proposal.length === 8;
      const editTarget = current.edit_target || (hasProposal ? "proposal" : "gt");
      const coords =
        hasProposal && editTarget === "proposal" ? current.proposal : current.gt;
      const target =
        hasProposal && editTarget === "proposal" ? "proposal" : "gt";
      window.__doccorner_drag_payload = JSON.stringify({
        name: current.name,
        target,
        coords,
        ts: Date.now(),
      });
      applyBtn.click();
    }

    canvas.addEventListener("pointerdown", (e) => {
      const { x, y } = eventPos(e);
      const idx = pickCorner(x, y);
      if (idx < 0) return;
      dragging = true;
      dragCorner = idx;
      canvas.setPointerCapture(e.pointerId);
      canvas.style.cursor = "grabbing";
      draw();
      e.preventDefault();
    });

    canvas.addEventListener("pointermove", (e) => {
      if (!dragging || dragCorner < 0 || !current) return;
      const hasProposal = Array.isArray(current.proposal) && current.proposal.length === 8;
      const editTarget = current.edit_target || (hasProposal ? "proposal" : "gt");
      const coords =
        hasProposal && editTarget === "proposal" ? current.proposal : current.gt;
      if (!Array.isArray(coords) || coords.length !== 8) return;
      const { x, y, w, h } = eventPos(e);
      coords[dragCorner * 2] = clamp(x / w, 0, 1);
      coords[dragCorner * 2 + 1] = clamp(y / h, 0, 1);
      draw();
      e.preventDefault();
    });

    canvas.addEventListener("pointerup", (e) => {
      if (!dragging) return;
      dragging = false;
      dragCorner = -1;
      canvas.style.cursor = "crosshair";
      sendUpdate();
      draw();
      e.preventDefault();
    });

    canvas.addEventListener("pointercancel", () => {
      dragging = false;
      dragCorner = -1;
      canvas.style.cursor = "crosshair";
      draw();
    });

    window.addEventListener("resize", draw);

    const observerTick = () => {
      const v = stateEl.value || "";
      if (v !== lastStateStr) {
        lastStateStr = v;
        try {
          current = v ? JSON.parse(v) : null;
        } catch {
          current = null;
        }
        const imgNow = container.querySelector("img");
        if (imgNow && imgNow !== img) {
          img = imgNow;
          host = img.parentElement;
          if (host) {
            host.style.position = "relative";
            if (canvas.parentElement !== host) host.appendChild(canvas);
          }
          img.addEventListener("load", draw, { once: true });
        }
        draw();
      }
    };

    setInterval(observerTick, 100);
    img.addEventListener("load", draw, { once: true });
    observerTick();
    return true;
  }

  const ensure = () => {
    if (!setup()) setTimeout(ensure, 300);
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", ensure);
  } else {
    ensure();
  }
})();
"""


def main():
    parser = argparse.ArgumentParser(description="DocCornerNet Outlier Review App")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--purple-checkpoint",
        type=str,
        default=DEFAULT_PURPLE_CHECKPOINT,
        help=f"Checkpoint used for the purple overlay bbox (default: {DEFAULT_PURPLE_CHECKPOINT})",
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=str,
        default=DEFAULT_CANDIDATE_CHECKPOINT,
        help=f"Candidate checkpoint used for proposals (default: {DEFAULT_CANDIDATE_CHECKPOINT})",
    )
    parser.add_argument(
        "--teacher-onnx",
        type=str,
        default=DEFAULT_TEACHER_ONNX,
        help=f"Teacher ONNX used for proposals (default: {DEFAULT_TEACHER_ONNX})",
    )
    parser.add_argument(
        "--teacher-input-norm",
        type=str,
        default=DEFAULT_TEACHER_INPUT_NORM,
        choices=["imagenet", "zero_one", "raw255", "m1p1"],
        help="Normalization expected by the teacher ONNX input.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split to review (default: val)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run server on (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public link (for remote access)",
    )

    args = parser.parse_args()

    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")

    # Create app
    app = OutlierReviewApp(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        purple_checkpoint_path=args.purple_checkpoint,
        candidate_checkpoint_path=args.candidate_checkpoint,
        teacher_onnx_path=args.teacher_onnx,
        teacher_input_norm=args.teacher_input_norm,
    )

    # Create and launch Gradio interface
    demo = create_app(app)
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=CUSTOM_CSS,
        js=CUSTOM_JS,
    )


if __name__ == "__main__":
    main()
