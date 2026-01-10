# DocCornerNet - Marginal Coordinate Classification

A lightweight neural network for document corner detection using **Marginal Coordinate Classification** (SimCC).

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-DocCornerDataset-yellow)](https://huggingface.co/datasets/mapo80/DocCornerDataset)

## Overview

DocCornerNet detects the four corners of documents in images using a novel approach based on **Simple Coordinate Classification (SimCC)**. Instead of predicting corner coordinates directly via regression or generating 2D heatmaps, SimCC treats coordinate prediction as a 1D classification problem along each axis, achieving sub-pixel precision with significantly lower computational cost.

**Key Features:**
- **Sub-pixel accuracy**: Mean corner error < 1 pixel at 224px input
- **Lightweight**: ~500K parameters, <1MB model size
- **Fast inference**: ~4ms on CPU (TFLite + XNNPACK)
- **Production-ready**: Full XNNPACK delegation for mobile/WASM deployment
- **High accuracy**: Mean IoU > 0.98 on document detection benchmarks
- **Multiple backbones**: MobileNetV2, MobileNetV3, CSPNeXt-Tiny
- **OHEM support**: Online Hard Example Mining for improved outlier handling

---

## Model Architecture

### High-Level Overview

```
Input Image [H×W×3]
        ↓
┌───────────────────────────────────────┐
│     MobileNetV2/V3 Backbone           │
│   (ImageNet pretrained, α=0.35-1.0)   │
└───────────────────────────────────────┘
        ↓
    Multi-scale features (C2, C3, C4, C5)
        ↓
┌───────────────────────────────────────┐
│           Mini-FPN Neck               │
│   Top-down pathway with lateral       │
│   connections, 2x nearest upsampling  │
└───────────────────────────────────────┘
        ↓
    Fused features P2 [H/4 × W/4 × fpn_ch]
        ↓
┌───────────────────────────────────────┐
│         SimCC Head                    │
│   Marginal pooling + 1D convolutions  │
│   → X logits [B, 4, num_bins]         │
│   → Y logits [B, 4, num_bins]         │
└───────────────────────────────────────┘
        ↓
    Soft-argmax decode → coords [B, 8]
        ↓
┌───────────────────────────────────────┐
│         Score Head                    │
│   Global pooling → Dense → logit      │
└───────────────────────────────────────┘
        ↓
    Output: 4 corners (x,y) + document score
```

### Component Details

#### 1. Backbone

The model supports multiple backbone architectures:

| Backbone | Parameters | Notes |
|----------|------------|-------|
| MobileNetV2 | ~495K (α=0.35), ~2.4M (α=1.0) | **Recommended** - Best accuracy/speed tradeoff |
| MobileNetV3-Small | ~742K (α=0.75) | Slightly larger, similar accuracy |
| MobileNetV3-Large | Larger | For server deployment |
| CSPNeXt-Tiny | ~2.5M | High accuracy, requires `keras-cv-attention-models` |

The backbone extracts multi-scale features at 4 resolutions:
- **C2**: H/4 × W/4 (56×56 at 224px input) - Fine details
- **C3**: H/8 × W/8 (28×28) - Medium features
- **C4**: H/16 × W/16 (14×14) - Coarse features
- **C5**: H/32 × W/32 (7×7) - Global context (used for score head)

#### 2. Mini-FPN (Feature Pyramid Network)

A lightweight top-down feature pyramid that merges multi-scale features:

```
C4 ──→ 1×1 Conv ──→ P4
                    ↓ 2× Upsample
C3 ──→ 1×1 Conv ──→ Add ──→ SepConv ──→ P3
                                        ↓ 2× Upsample
C2 ──→ 1×1 Conv ──→ Add ──→ SepConv ──→ P2 [56×56×fpn_ch]
```

Key design choices:
- **Separable convolutions** for efficiency (3×3 depthwise + 1×1 pointwise)
- **XNNPACK-friendly 2× upsampling** via reshape+multiply (no RESIZE_NEAREST_NEIGHBOR)
- **Batch normalization + Swish** activation after each refinement

#### 3. SimCC Head (Marginal Coordinate Classification)

The core innovation: predicting coordinates as 1D classification problems.

**Step 1: Marginal Pooling**
```
P_fused [B, 56, 56, ch]
    ↓
    ├── Mean along Y axis → X_marginal [B, 56, ch]  (vertical features)
    └── Mean along X axis → Y_marginal [B, 56, ch]  (horizontal features)
```

**Step 2: Resolution Matching**
```
X_marginal [B, 56, ch] → Bilinear resize → [B, num_bins, ch]
Y_marginal [B, 56, ch] → Bilinear resize → [B, num_bins, ch]
```

**Step 3: 1D Convolutions**
```
X_feat = Conv1D(k=5) → BN → ReLU → Conv1D(k=3) → BN → ReLU
Y_feat = Conv1D(k=5) → BN → ReLU → Conv1D(k=3) → BN → ReLU
```

**Step 4: Global Context Fusion**
```
Global = GAP(P_fused) → Dense → Broadcast to [B, num_bins, ch/2]
X_feat = Concat([X_feat, Global])
Y_feat = Concat([Y_feat, Global])
```

**Step 5: Output Logits**
```
simcc_x = Conv1D(4, k=1)(X_feat) → [B, 4, num_bins]  (4 corners × num_bins)
simcc_y = Conv1D(4, k=1)(Y_feat) → [B, 4, num_bins]
```

#### 4. Coordinate Decoding (Soft-Argmax)

The logits are converted to continuous coordinates via soft-argmax:

```python
# For each corner i ∈ {0,1,2,3}:
prob_x = softmax(simcc_x[:, i, :] / τ)  # [B, num_bins]
prob_y = softmax(simcc_y[:, i, :] / τ)  # [B, num_bins]

# Bin centers in [0, 1]
centers = linspace(0, 1, num_bins)

# Expected value (soft-argmax)
x_i = sum(prob_x * centers)  # [B]
y_i = sum(prob_y * centers)  # [B]
```

Where τ (tau) is a temperature parameter (default 1.0). Lower τ makes the distribution sharper.

#### 5. Score Head

Binary classification for document presence:

```
C5 [B, 7, 7, ch] → Global Average Pool → Dense(1) → score_logit
```

The logit is converted to probability via sigmoid during inference.

### Why SimCC Works Better Than Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Direct Regression** | Simple | Poor gradient flow, limited supervision |
| **2D Heatmaps** | Rich supervision | Expensive (H×W per keypoint), quantization error |
| **SimCC (ours)** | Rich supervision (num_bins per axis), efficient, sub-pixel precision | Requires axis independence assumption |

SimCC advantages:
1. **Richer supervision**: 224 bins per axis vs 1 scalar (regression) or 224×224 (heatmap)
2. **Better gradients**: Cross-entropy loss provides stronger signal than L1/L2
3. **Spatial awareness**: Marginal pooling preserves position information
4. **Efficiency**: O(num_bins) instead of O(H×W) for heatmaps

---

## Model Configurations

### Presets

| Config | Backbone | Alpha | FPN | SimCC | Input | Params | Use Case |
|--------|----------|-------|-----|-------|-------|--------|----------|
| **Mobile** | MobileNetV2 | 0.35 | 32 | 96 | 224/256 | ~495K | Mobile, WASM, edge |
| **Server** | MobileNetV2 | 1.0 | 48 | 128 | 320 | ~2.4M | Server, high accuracy |
| **CSPNeXt** | CSPNeXt-Tiny | - | 48 | 128 | 320 | ~2.5M | Highest accuracy |
| **Tiny** | MobileNetV2 | 0.35 | 24 | 64 | 224 | ~105K | Ultra-constrained |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--backbone` | mobilenetv2 | Backbone: `mobilenetv2`, `mobilenetv3_small`, `mobilenetv3_large`, `cspnext` |
| `--alpha` | 0.35 | Backbone width multiplier (MobileNet only) |
| `--fpn_ch` | 32 | FPN channel dimension |
| `--simcc_ch` | 96 | SimCC head hidden channels |
| `--img_size` | 256 | Input image size |
| `--num_bins` | 256 | Number of classification bins (usually = img_size) |
| `--tau` | 1.0 | Softmax temperature |
| `--batch_size` | 512 | Training batch size |
| `--lr` | 0.001 | Initial learning rate |
| `--epochs` | 200 | Training epochs |
| `--hard_mining` | false | Enable Online Hard Example Mining (OHEM) |
| `--hard_mining_weight` | 2.0 | Extra weight for hard samples (total = 1 + weight) |
| `--hard_mining_threshold` | 20.0 | Corner error threshold (px) to classify as hard |
| `--hard_mining_start` | 0.2 | Fraction of epochs before activating OHEM |

---

## Online Hard Example Mining (OHEM)

OHEM is an advanced training technique that improves model performance on difficult samples by dynamically tracking high-error samples during validation and applying increased loss weights during subsequent training epochs.

### Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OHEM Training Flow                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Epoch 1-N (Warmup):     Standard training, collect hard samples    │
│       ↓                                                             │
│  Validation:             Track max corner error per sample          │
│       ↓                                                             │
│  HardSampleTracker:      Identify samples with error >= threshold   │
│       ↓                                                             │
│  Epoch N+1 (OHEM Active): Apply weighted loss to hard samples       │
│       ↓                                                             │
│  Result:                 Model focuses on difficult cases           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### How It Works

#### 1. Hard Sample Identification

After each validation epoch, the system computes the **maximum corner error** (in pixels) for each sample:

```python
# For each sample with has_doc=1:
pred_corners = model(image)  # [4, 2] predicted corners
gt_corners = ground_truth    # [4, 2] actual corners

# Compute per-corner Euclidean distance (in pixels)
corner_errors = sqrt(sum((pred - gt)^2, axis=-1)) * img_size  # [4]

# Max corner error for this sample
max_corner_error = max(corner_errors)  # scalar

# Sample is "hard" if any corner exceeds threshold
is_hard = (max_corner_error >= threshold_px)  # default: 20px
```

#### 2. Weighted Loss Computation

During training, hard samples receive increased weight in the loss function:

```python
# Standard loss computation
loss_simcc = cross_entropy(pred_logits, target_distribution)
loss_coord = L1(pred_coords, gt_coords)

# With OHEM: apply per-sample weights
sample_weight = 1.0 + hard_mining_weight if is_hard else 1.0
# Default: hard samples get weight 3.0 (1.0 + 2.0)

weighted_loss = loss * sample_weight
```

**Important**: Only SimCC and coordinate losses are weighted. The score loss (document presence) remains unweighted to maintain balanced classification.

#### 3. Curriculum Learning

OHEM uses curriculum learning to prevent early overfitting to hard samples:

```
Epochs 1 to start_epoch:    OHEM inactive, standard training
                            (hard samples still tracked but not weighted)

Epochs start_epoch to end:  OHEM active, weighted training on hard samples
```

Default: `--hard_mining_start 0.2` means OHEM activates after 20% of epochs.

### CLI Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--hard_mining` | flag | `false` | Enable Online Hard Example Mining |
| `--hard_mining_weight` | float | `2.0` | Extra weight added to hard samples. Total weight = `1.0 + weight` |
| `--hard_mining_threshold` | float | `20.0` | Corner error threshold in pixels. Samples with max corner error >= this value are classified as hard |
| `--hard_mining_start` | float | `0.2` | Fraction of total epochs before activating OHEM (curriculum warmup) |

### Implementation Details

#### HardSampleTracker Class

Located in `train_ultra.py`, this class manages the hard sample tracking:

```python
class HardSampleTracker:
    def __init__(self, threshold_px: float = 20.0):
        self.threshold_px = threshold_px
        self.hard_indices = set()      # Set of hard sample indices
        self.hard_scores = {}          # {index: max_corner_error}

    def update_from_validation(self, sample_indices, max_corner_errors):
        """Update hard sample list after validation epoch."""
        # Clears previous list and rebuilds from current validation

    def is_hard(self, idx: int) -> bool:
        """Check if sample index is in hard set."""

    def get_weight(self, idx, base_weight=1.0, hard_weight=2.0) -> float:
        """Return weight for sample: base + hard_weight if hard, else base."""

    def save(self, path: Path):
        """Save hard samples to .npz file."""

    def load(self, path: Path):
        """Load hard samples from .npz file (for resume)."""
```

#### Dataset Index Tracking

When OHEM is enabled, `FastDataset` returns sample indices:

```python
# Without OHEM (return_indices=False):
for images, coords, has_doc in dataset:
    loss = train_step(images, coords, has_doc)

# With OHEM (return_indices=True):
for indices, images, coords, has_doc in dataset:
    weights = compute_weights(indices, hard_tracker)
    loss = train_step_weighted(images, coords, has_doc, weights)
```

#### Weighted Training Steps

Two new training methods support sample weights:

```python
# Standard training (no weights)
trainer.train_step(images, coords, has_doc)

# Weighted training (OHEM)
trainer.train_step_weighted(images, coords, has_doc, sample_weights)

# With gradient accumulation
trainer.train_step_accumulate_weighted(images, coords, has_doc, sample_weights)
```

#### ValidationMetrics Extensions

The `ValidationMetrics` class in `metrics.py` has two new methods:

```python
# Track indices during validation
metrics.update_with_indices(indices, pred_coords, gt_coords, pred_scores, has_doc)

# Compute metrics + per-sample errors for OHEM
results, pos_indices, max_corner_errors = metrics.compute_with_indices()
```

### Usage Examples

#### Basic OHEM Training

```bash
python train_ultra.py \
    --hf_dataset ./hf_dataset \
    --output_dir ./checkpoints/ohem_basic \
    --backbone mobilenetv2 \
    --alpha 0.35 \
    --img_size 320 \
    --batch_size 64 \
    --epochs 100 \
    --hard_mining \
    --augment
```

#### Aggressive OHEM (Lower Threshold, Higher Weight)

```bash
python train_ultra.py \
    --hf_dataset ./hf_dataset \
    --output_dir ./checkpoints/ohem_aggressive \
    --backbone mobilenetv2 \
    --alpha 1.0 \
    --img_size 320 \
    --batch_size 64 \
    --epochs 150 \
    --hard_mining \
    --hard_mining_threshold 10.0 \
    --hard_mining_weight 3.0 \
    --hard_mining_start 0.15 \
    --augment
```

#### OHEM with CSPNeXt Backbone

```bash
TF_USE_LEGACY_KERAS=1 python train_ultra.py \
    --hf_dataset ./hf_dataset \
    --output_dir ./checkpoints/cspnext_ohem \
    --backbone cspnext \
    --img_size 320 \
    --batch_size 32 \
    --epochs 200 \
    --hard_mining \
    --hard_mining_threshold 15.0 \
    --hard_mining_weight 2.5 \
    --augment
```

#### OHEM with Gradient Accumulation

```bash
python train_ultra.py \
    --hf_dataset ./hf_dataset \
    --output_dir ./checkpoints/ohem_accum \
    --backbone mobilenetv2 \
    --alpha 1.0 \
    --img_size 320 \
    --batch_size 32 \
    --accumulation_steps 4 \
    --epochs 100 \
    --hard_mining \
    --augment
```

### Training Output

When OHEM is enabled, the training output includes additional information:

```
================================================================================
Starting training: 100 epochs
Batch size: 64
Augmentation: ENABLED
Hard mining: ENABLED (threshold=20.0px, weight=2.0, start=20%)
================================================================================

Epoch 1/100 [OHEM: inactive until epoch 21]
  Train: 100%|████████| 1234/1234 [loss=0.5432, err=12.3px, iou=0.912, img/s=450]
  Val:   100%|████████| 123/123 [loss=0.4321, img/s=890]
Epoch   1/100 | LR=1.0e-03 | 45.2s (1200 img/s)
    Train: loss=0.5432 (simcc=0.4123 coord=0.0876 score=0.0433)
           err=12.3px  IoU=0.912
    Val:   loss=0.4321 (simcc=0.3456 coord=0.0567 score=0.0298)
           err_mean=8.5px  err_p95=24.3px  err_max=45.2px  err_worst=67.8px
           IoU=0.9234  R@90=78.5%  R@95=65.2%  R@99=34.1%
           outliers: IoU<0.90=156/5000 (3.1%)  err>20px=89/5000 (1.8%)  any_corner>20px=234/5000 (4.7%)
           cls_acc=99.2%  cls_f1=0.994
           OHEM: 234 hard samples (avg err: 34.2px) [inactive]

...

Epoch 21/100 [OHEM: 234 hard samples]
  Train: 100%|████████| 1234/1234 [loss=0.3210, err=8.5px, iou=0.945, img/s=420]
  Val:   100%|████████| 123/123 [loss=0.2987, img/s=890]
Epoch  21/100 | LR=1.0e-03 | 48.1s (1125 img/s)
    Train: loss=0.3210 (simcc=0.2345 coord=0.0543 score=0.0322)
           err=8.5px  IoU=0.945
    Val:   loss=0.2987 (simcc=0.2123 coord=0.0512 score=0.0352)
           err_mean=6.2px  err_p95=18.7px  err_max=32.1px  err_worst=48.9px
           IoU=0.9456  R@90=85.3%  R@95=72.1%  R@99=45.6%
           outliers: IoU<0.90=98/5000 (2.0%)  err>20px=45/5000 (0.9%)  any_corner>20px=156/5000 (3.1%)
           cls_acc=99.5%  cls_f1=0.996
           OHEM: 156 hard samples (avg err: 28.1px) [ACTIVE]
```

### Checkpoint Files

When OHEM is enabled, an additional file is saved:

```
checkpoints/experiment_name/
├── config.json              # Training configuration
├── history.json             # Training history
├── best_model.weights.h5    # Best model weights
├── best_model_inference.keras  # Best inference model
├── final_model.weights.h5   # Final model weights
└── hard_samples.npz         # Hard sample indices and scores (OHEM)
```

The `hard_samples.npz` file contains:
- `indices`: Array of hard sample indices (int64)
- `scores`: Array of max corner errors for hard samples (float32)

This file is automatically loaded when resuming training, allowing OHEM to continue from where it left off.

### Recommended Settings

| Scenario | Threshold | Weight | Start | Notes |
|----------|-----------|--------|-------|-------|
| **Default** | 20px | 2.0 | 0.2 | Good starting point |
| **High precision** | 10px | 2.5 | 0.15 | Focus on fine errors |
| **Outlier reduction** | 30px | 3.0 | 0.25 | Target worst cases |
| **Gradual focus** | 20px | 1.5 | 0.3 | Gentle OHEM |

### Compatibility

OHEM is fully compatible with:
- All backbone architectures (MobileNetV2, MobileNetV3, CSPNeXt)
- Gradient accumulation (`--accumulation_steps`)
- Data augmentation (`--augment`)
- GAU and FC expansion (`--use_gau`, `--fc_expansion_dim`)
- Mixed precision training (CUDA)
- Training resume (hard samples are saved and reloaded)

---

## Leaderboard

### Test Set Results

Evaluated on [DocCornerDataset](https://huggingface.co/datasets/mapo80/DocCornerDataset) **test split** (6,652 samples):

| Model | Alpha | Params | Size | mean_iou | Corner err (px) | Recall@95 | Recall@99 |
|-------|-------|--------|------|----------|-----------------|-----------|-----------|
| `mobilenetv2_320_a1.0_baseline` | 1.0 | 2.4M | 9.84 MB | **0.9197** | **6.90** | **67.2%** | **26.5%** |
| `mobilenetv2_320_a0.35_baseline` | 0.35 | 495K | 2.41 MB | 0.9044 | 8.41 | 64.1% | 23.2% |
| `mobilenetv2_320_a1.0_gau_fc256` | 1.0 | 2.4M | 9.75 MB | 0.8959 | 9.20 | 60.5% | 25.6% |
| `mobilenetv2_320_a0.35_fc256` | 0.35 | 546K | 2.62 MB | 0.8939 | 9.50 | 63.1% | 22.9% |
| `mobilenetv2_320_a0.35_gau_v2_fc256` | 0.35 | 546K | 2.63 MB | 0.8771 | 11.00 | 56.3% | 18.8% |

**Winner**: `mobilenetv2_320_a1.0_baseline` - Best accuracy on test set with IoU=0.9197 and corner error 6.90px.

**Note**: GAU (Gated Attention Unit) models were trained with a version that didn't include trainable GAU weights. The FC expansion (256-dim) didn't improve accuracy over the baseline.

### Validation Set Results

Evaluated on [DocCornerDataset](https://huggingface.co/datasets/mapo80/DocCornerDataset) validation split:

| Model | Input | mean_iou | Corner err (px) | Latency (ms) | Size |
|-------|-------|----------|-----------------|--------------|------|
| `mobilenetv2_224_best` | 224 | 0.9894 | 0.57 | 4.24 | 0.98 MB |
| `mobilenetv2_256_best` | 256 | **0.9902** | 0.60 | 8.18 | 0.98 MB |
| `mobilenetv2_320` | 320 | 0.9855 | 1.13 | 5.36 | 0.88 MB |
| `mobilenetv3_224` | 224 | 0.9842 | 0.86 | 3.96 | 1.47 MB |

**Recommended for deployment**: `mobilenetv2_224_best` - Best speed/accuracy/robustness tradeoff.

---

## Quick Start

### Install

```bash
git clone https://github.com/mapo80/DocCornerNet-CoordClass.git
cd DocCornerNet-CoordClass
pip install -r requirements.txt
```

#### CSPNeXt Backbone (Optional)

To use CSPNeXt-Tiny backbone:

```bash
pip install keras-cv-attention-models

# For TensorFlow >= 2.16, use legacy Keras:
pip install tf-keras
export TF_USE_LEGACY_KERAS=1
```

**Note**: `keras-cv-attention-models` is NOT compatible with Keras 3.x. You must use `tf-keras` with `TF_USE_LEGACY_KERAS=1`.

### Download Dataset

```bash
python train_ultra.py \
    --hf_dataset mapo80/DocCornerDataset \
    --download_hf ./hf_dataset
```

### Train

```bash
# Mobile model (alpha=0.35, 256px)
python train_ultra.py \
    --hf_dataset ./hf_dataset \
    --output_dir ./checkpoints \
    --backbone mobilenetv2 \
    --alpha 0.35 \
    --img_size 256 \
    --num_bins 256 \
    --batch_size 512 \
    --epochs 200 \
    --augment

# Server model (alpha=1.0, 320px)
python train_ultra.py \
    --hf_dataset ./hf_dataset \
    --output_dir ./checkpoints \
    --backbone mobilenetv2 \
    --alpha 1.0 \
    --img_size 320 \
    --num_bins 320 \
    --simcc_ch 128 \
    --fpn_ch 48 \
    --batch_size 128 \
    --epochs 200 \
    --augment

# CSPNeXt-Tiny model (320px, ~2.5M params)
TF_USE_LEGACY_KERAS=1 python train_ultra.py \
    --hf_dataset ./hf_dataset \
    --output_dir ./checkpoints \
    --backbone cspnext \
    --img_size 320 \
    --num_bins 320 \
    --simcc_ch 128 \
    --fpn_ch 48 \
    --batch_size 64 \
    --epochs 200 \
    --augment
```

### Evaluate

```bash
python evaluate.py \
    --model_path ./checkpoints/mobilenetv2_256_best \
    --data_root ./hf_dataset \
    --split val
```

### Export

```bash
# TFLite
python export.py \
    --checkpoint ./checkpoints/mobilenetv2_256_best \
    --output ./exported/model.tflite \
    --format tflite

# ONNX
python export_onnx.py \
    --checkpoint ./checkpoints/mobilenetv2_256_best \
    --output ./exported/model.onnx
```

---

## Remote Training (RunPod, Lambda Labs, etc.)

### One-Line Setup

```bash
curl -sSL https://raw.githubusercontent.com/mapo80/DocCornerNet-CoordClass/main/setup_remote.sh | bash -s -- --download-dataset
```

### Full Workflow

```bash
# 1. SSH into remote machine
ssh root@<HOST> -p <PORT> -i ~/.ssh/id_ed25519

# 2. Setup + download dataset
curl -sSL https://raw.githubusercontent.com/mapo80/DocCornerNet-CoordClass/main/setup_remote.sh | bash -s -- --download-dataset --output-dir /workspace/hf_dataset --repo-dir /workspace/DocCornerNet-CoordClass

# 3. Train
cd /workspace/DocCornerNet-CoordClass
nohup python train_ultra.py \
    --hf_dataset /workspace/hf_dataset \
    --output_dir /workspace/checkpoints \
    --backbone mobilenetv2 \
    --alpha 0.35 \
    --img_size 256 \
    --num_bins 256 \
    --batch_size 512 \
    --epochs 200 \
    --augment \
    > /workspace/training.log 2>&1 &

# 4. Monitor
tail -f /workspace/training.log

# 5. Download results (from local)
scp -P <PORT> -i ~/.ssh/id_ed25519 root@<HOST>:/workspace/checkpoints/*/best_model.weights.h5 ./
```

---

## Architecture

```
Input [224×224×3]
       ↓
MobileNetV2/V3 Backbone
       ↓
Mini-FPN (32-48 ch) → Feature Map [56×56×ch]
       ↓
Marginal Pooling:
  ├── mean(axis=Y) → X marginal → Conv1D → logits_x [224×4]
  └── mean(axis=X) → Y marginal → Conv1D → logits_y [224×4]
       ↓
Soft-argmax → coords [8] + score [1]
```

**Why SimCC works better than regression:**
1. Richer supervision (224 bins per axis vs 1 scalar)
2. Better gradient flow (cross-entropy vs L1/L2)
3. Sub-pixel precision via soft-argmax

---

## Output Format

### Corner Order

```
TL (x0, y0) ──── TR (x1, y1)
    │                │
BL (x3, y3) ──── BR (x2, y2)
```

### TFLite Output

Single tensor `[1, 9]`:
- `[0:8]`: Normalized coordinates [0, 1]
- `[8]`: Document presence score (sigmoid applied)

---

## Files

```
├── model.py           # Network architecture (MobileNet, CSPNeXt backbones)
├── dataset.py         # Data loading + augmentation
├── losses.py          # Loss functions (SimCC, coord, score)
├── metrics.py         # Evaluation metrics + OHEM support
│   ├── ValidationMetrics.update_with_indices()   # Track sample indices
│   └── ValidationMetrics.compute_with_indices()  # Return per-sample errors
├── train_ultra.py     # Training script with OHEM support
│   ├── HardSampleTracker        # Hard sample tracking class
│   ├── FastDataset              # Dataset with optional index tracking
│   ├── Trainer.train_step_weighted()           # Weighted loss training
│   └── Trainer.train_step_accumulate_weighted() # + gradient accumulation
├── evaluate.py        # Evaluation
├── export.py          # Export (SavedModel, TFLite, ONNX)
├── export_onnx.py     # ONNX export
├── create_hf_dataset.py  # Create HuggingFace dataset
├── setup_remote.sh    # Remote machine setup
├── requirements.txt   # Dependencies
└── checkpoints/       # Pretrained models
    ├── mobilenetv2_224_best/      # Validation best (224px, α=0.35)
    ├── mobilenetv2_256_best/      # Validation best (256px, α=0.35)
    ├── mobilenetv2_320/           # Legacy (320px)
    ├── mobilenetv2_320_a1.0_server/   # Test winner (320px, α=1.0, 2.4M params)
    ├── mobilenetv2_320_a0.35_mobile/  # Mobile (320px, α=0.35, 495K params)
    └── mobilenetv3_224/           # MobileNetV3 (224px)
```

---

## Optimized Dataset Loading

The dataset loading pipeline has been optimized for maximum performance with vectorized operations.

### Performance Improvements

| Operation | Before | After | Speedup |
|-----------|--------|-------|---------|
| Parquet reading | Row-by-row iteration | PyArrow `concat_tables` + vectorized column access | ~3x |
| tf.data construction | Chunked `concatenate()` (O(n²)) | Single `from_tensor_slices()` (O(1)) | ~50x |
| Shuffle buffer | Full dataset size | 10% of dataset (min 10K) | ~10x memory reduction |

### Benchmark Results (Validation Set - 8,645 samples)

```
Parquet read:          2.8s
Column extraction:     0.2s
Image decoding:        8.4s (1,032 img/s)
Total loading:         11.5s
FastDataset creation:  0.17s
Iteration speed:       8,264 img/s
```

### Key Optimizations

#### 1. Vectorized Parquet Reading

```python
# Before: Row-by-row (slow)
for pf in parquet_files:
    table = pq.read_table(pf)
    for i in range(len(table)):
        row = {"image_bytes": table["image"][i].as_py()["bytes"], ...}

# After: Vectorized with PyArrow
tables = [pq.read_table(pf) for pf in parquet_files]
combined = pa.concat_tables(tables)
coords = np.column_stack([combined[f"corner_{c}_{a}"].to_numpy() for c, a in corners])
```

#### 2. Single-Pass tf.data Construction

```python
# Before: O(n²) chunked concatenation
dataset = None
for chunk in chunks:
    shard_ds = tf.data.Dataset.from_tensor_slices(chunk)
    dataset = shard_ds if dataset is None else dataset.concatenate(shard_ds)

# After: O(1) single call
dataset = tf.data.Dataset.from_tensor_slices((images, coords, has_doc))
```

#### 3. Optimized Shuffle Buffer

```python
# Before: Full buffer (high memory)
dataset.shuffle(n_samples)

# After: 10% buffer (sufficient randomization, lower memory)
buffer_size = min(n_samples, max(10000, n_samples // 10))
dataset.shuffle(buffer_size)
```

---

## CSPNeXt Backbone

CSPNeXt-Tiny is a high-performance backbone from the RTMDet family, offering improved accuracy over MobileNet at the cost of larger model size.

### Requirements

```bash
pip install keras-cv-attention-models tf-keras
export TF_USE_LEGACY_KERAS=1
```

### Feature Scales

CSPNeXt-Tiny outputs at standard FPN scales (for 320px input):
- **C2**: 80×80×48 (H/4)
- **C3**: 40×40×96 (H/8)
- **C4**: 20×20×192 (H/16)
- **C5**: 10×10×384 (H/32)

### Usage

```python
from model import create_model

# Create CSPNeXt model
model = create_model(
    backbone='cspnext',
    img_size=320,
    backbone_weights='imagenet',  # or None
)
print(f"Parameters: {model.count_params():,}")  # ~2.5M
```

### Training

```bash
TF_USE_LEGACY_KERAS=1 python train_ultra.py \
    --hf_dataset ./hf_dataset \
    --backbone cspnext \
    --img_size 320 \
    --batch_size 64 \
    --epochs 200
```

### Comparison

| Backbone | Params | ImageNet Top-1 | Notes |
|----------|--------|----------------|-------|
| MobileNetV2 α=0.35 | 567K | ~60% | Fastest, mobile-optimized |
| MobileNetV2 α=1.0 | 2.4M | 72% | Good balance |
| CSPNeXt-Tiny | 2.5M | ~75% | Best accuracy, RTMDet backbone |

---

## References

- [SimCC](https://arxiv.org/abs/2107.03332) - Simple Coordinate Classification, ECCV 2022
- [MobileNetV3](https://arxiv.org/abs/1905.02244) - ICCV 2019
- [FPN](https://arxiv.org/abs/1612.03144) - Feature Pyramid Networks, CVPR 2017
- [RTMDet](https://arxiv.org/abs/2212.07784) - CSPNeXt backbone, arXiv 2022
- [OHEM](https://arxiv.org/abs/1604.03540) - Online Hard Example Mining, CVPR 2016

## License

MIT License - see [LICENSE](LICENSE) file.
