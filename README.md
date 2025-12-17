# DocCornerNet - Marginal Coordinate Classification

A lightweight neural network for document corner detection using a novel **Marginal Coordinate Classification** approach.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-DocCornerDataset-yellow)](https://huggingface.co/datasets/mapo80/DocCornerDataset)

## Results

| Metric | Value |
|--------|-------|
| **Mean IoU** | 0.9820 |
| **Median IoU** | 0.9881 |
| **Corner Error (mean)** | 0.94 px |
| **Corner Error (p95)** | 2.07 px |
| **Recall@50** | 99.89% |
| **Recall@75** | 99.57% |
| **Recall@90** | 98.81% |
| **Recall@95** | 96.82% |
| **Classification Accuracy** | 99.95% |
| **Classification F1** | 99.97% |
| **Parameters** | < 1M |

## Pretrained Models

Pretrained models are included in the `checkpoints/` directory:

### Model Files

| File | Description | Size |
|------|-------------|------|
| `best_model.keras` | Best model (full, for training resume) | 3.5 MB |
| `best_model.weights.h5` | Best model weights only | 3.3 MB |
| `best_model_inference.keras` | Best model (inference optimized) | 3.5 MB |
| `final_model.keras` | Final epoch model (full) | 3.5 MB |
| `final_model.weights.h5` | Final epoch weights only | 3.3 MB |
| `final_model_inference.keras` | Final model (inference optimized) | 3.5 MB |
| `config.json` | Training configuration | 729 B |
| `history.json` | Training history (loss, metrics per epoch) | 90 KB |

### Quick Start with Pretrained Models

```python
import tensorflow as tf
from model import create_model, create_inference_model

# Load the best model for inference
model = tf.keras.models.load_model("checkpoints/best_model_inference.keras")

# Or load weights into a new model
model = create_inference_model(img_size=224, alpha=0.75, fpn_ch=48, simcc_ch=128)
model.load_weights("checkpoints/best_model.weights.h5")

# Inference
# Input: [B, 224, 224, 3] RGB image (ImageNet normalized)
# Output: coords [B, 8], score_logit [B, 1]
coords, score_logit = model(image_tensor)
score = tf.nn.sigmoid(score_logit)  # Apply sigmoid for probability
```

### Training Configuration

The pretrained models were trained with:

| Parameter | Value |
|-----------|-------|
| `alpha` | 0.75 |
| `fpn_ch` | 48 |
| `simcc_ch` | 128 |
| `img_size` | 224 |
| `sigma_px` | 3.0 |
| `batch_size` | 64 |
| `epochs` | 100 |
| `lr` | 0.0002 |
| `w_simcc` | 1.0 |
| `w_coord` | 0.5 |
| `w_score` | 0.5 |

## Dataset

This model is trained on the **DocCornerDataset** available on HuggingFace:

**[mapo80/DocCornerDataset](https://huggingface.co/datasets/mapo80/DocCornerDataset)**

### Download

```bash
# Using huggingface_hub
pip install huggingface_hub
huggingface-cli download mapo80/DocCornerDataset --repo-type dataset --local-dir ./data
```

## Architecture

DocCornerNet uses **Marginal Coordinate Classification** combining:

1. **MobileNetV3Small** backbone (α=0.75, ImageNet pretrained)
2. **Mini-FPN** with 48 channels
3. **Marginal Pooling**: 2D features → separate 1D distributions for X/Y
4. **Conv1D Refinement** along each axis
5. **SimCC Loss** for soft coordinate classification

### Key Innovation: Marginal Pooling

Instead of GAP→FC (loses spatial info), we project features to 1D marginals:

```
Feature Map [56×56×48]
      │
      ├── mean(axis=Y) → X marginal [56] → upsample [224] → Conv1D → logits_x
      │
      └── mean(axis=X) → Y marginal [56] → upsample [224] → Conv1D → logits_y
```

**Why it works:**
- Each X position "sees" the sum of all rows → knows where vertical edges are
- Each Y position "sees" the sum of all columns → knows where horizontal edges are
- X and Y coordinates are predicted independently → less confusion
- Conv1D captures local patterns along each axis → sub-pixel precision

## Installation

```bash
git clone https://github.com/mapo80/DocCornerNet-CoordClass.git
cd DocCornerNet-CoordClass
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py \
    --data_root /path/to/dataset \
    --batch_size 64 \
    --epochs 100 \
    --lr 0.0002 \
    --augment \
    --cache_images \
    --fast_mode \
    --output_dir ./checkpoints
```

**Training Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_root` | required | Path to dataset |
| `--batch_size` | 32 | Batch size |
| `--epochs` | 100 | Training epochs |
| `--lr` | 2e-4 | Learning rate |
| `--alpha` | 0.75 | MobileNetV3 width multiplier |
| `--fpn_ch` | 48 | FPN channels |
| `--simcc_ch` | 128 | SimCC head channels |
| `--img_size` | 224 | Input image size |
| `--sigma_px` | 3.0 | Gaussian sigma for SimCC targets |
| `--augment` | flag | Enable data augmentation |
| `--cache_images` | flag | Pre-load images to RAM |
| `--fast_mode` | flag | GPU-accelerated augmentations |
| `--warmup_epochs` | 5 | LR warmup epochs |
| `--patience` | 15 | Early stopping patience |

### Evaluation

```bash
python evaluate.py \
    --model_path ./checkpoints/best_model.keras \
    --data_root /path/to/dataset \
    --split val \
    --batch_size 32
```

### Export

```bash
# Export to SavedModel and TFLite
python export.py \
    --weights ./checkpoints/best_model.weights.h5 \
    --output_dir ./exported \
    --format savedmodel tflite

# Export with INT8 quantization
python export.py \
    --weights ./checkpoints/best_model.weights.h5 \
    --output_dir ./exported \
    --format tflite_int8 \
    --representative_data /path/to/calibration/images
```

### TFLite Evaluation

```bash
python eval_tflite.py \
    --model_path ./exported/model_float32.tflite \
    --data_root /path/to/dataset
```

### TFLite V2-Compatible Export

The `convert_v2_compatible.py` script converts the model to TFLite with an output format **identical to [DocCornerNet-Regression](https://github.com/mapo80/DocCornerNet-Regression)**:

```bash
# First export to SavedModel
python export.py --weights ./checkpoints/best_model.weights.h5 --output_dir ./exported --format savedmodel

# Then convert to V2-compatible TFLite
cd exported
python convert_v2_compatible.py
```

**V2-Compatible Format:**

| Tensor | Name | Shape | Description |
|--------|------|-------|-------------|
| Input | `input` | `[1, 224, 224, 3]` | RGB image (float32) |
| Output | `Identity` | `[1, 9]` | 8 coords + sigmoid(score) |

**Output format:** `[x0, y0, x1, y1, x2, y2, x3, y3, score]`
- Coordinates: normalized [0, 1]
- Score: sigmoid probability [0, 1] (already applied)

This format is interchangeable with [DocCornerNet-Regression](https://github.com/mapo80/DocCornerNet-Regression) TFLite models.

**Generated files:**
- `model_float32.tflite` - Full precision
- `model_float16.tflite` - Half precision (smaller size)

## Output Format

### Corner Order

```
TL (x0, y0) ──── TR (x1, y1)
    │                │
    │                │
BL (x3, y3) ──── BR (x2, y2)

coords = [x0, y0, x1, y1, x2, y2, x3, y3]  # normalized [0, 1]
```

### Model Output

- `coords`: [B, 8] - Normalized corner coordinates
- `score`: [B, 1] - Document presence probability (apply sigmoid)

## Training Configuration

Configuration used for the reported results:

| Parameter | Value |
|-----------|-------|
| Input Size | 224×224 |
| Backbone | MobileNetV3Small (α=0.75) |
| FPN Channels | 48 |
| SimCC Channels | 128 |
| Batch Size | 64 |
| Epochs | 100 |
| Learning Rate | 2e-4 |
| Weight Decay | 1e-4 |
| Warmup Epochs | 5 |
| Loss Weights | SimCC=1.0, Coord=0.5, Score=0.5 |

## Files

```
├── checkpoints/
│   ├── best_model.keras           # Best model (full)
│   ├── best_model.weights.h5      # Best model weights
│   ├── best_model_inference.keras # Best model (inference)
│   ├── final_model.keras          # Final epoch model
│   ├── final_model.weights.h5     # Final epoch weights
│   ├── final_model_inference.keras# Final model (inference)
│   ├── config.json                # Training configuration
│   └── history.json               # Training history
├── model.py                       # Network architecture
├── losses.py                      # Loss functions (SimCC, Coord, Score)
├── metrics.py                     # Evaluation metrics (IoU, corner error)
├── dataset.py                     # TF Dataset with augmentations
├── train.py                       # Training script
├── evaluate.py                    # Evaluation script
├── export.py                      # Export to SavedModel/TFLite/ONNX
├── eval_tflite.py                 # TFLite evaluation
├── convert_v2_compatible.py       # Convert to V2-compatible TFLite format
├── train_qat.py                   # Quantization-Aware Training
├── ptq_improved.py                # Post-Training Quantization
├── visualize_*.py                 # Augmentation visualization
└── README.md
```

## References

- [SimCC: A Simple Coordinate Classification Perspective for Human Pose Estimation](https://arxiv.org/abs/2107.03332) - ECCV 2022
- [MobileNetV3](https://arxiv.org/abs/1905.02244) - ICCV 2019
- [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144) - CVPR 2017

## License

MIT License - see [LICENSE](LICENSE) file.

## Citation

```bibtex
@misc{doccornernet2024,
  title={DocCornerNet: Marginal Coordinate Classification for Document Corner Detection},
  year={2024},
  url={https://github.com/mapo80/DocCornerNet-CoordClass}
}
```
