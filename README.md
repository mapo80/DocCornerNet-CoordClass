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
| MobileNetV2 | ~495K (α=0.35) | **Recommended** - Best accuracy/speed tradeoff |
| MobileNetV3-Small | ~742K (α=0.75) | Slightly larger, similar accuracy |
| MobileNetV3-Large | Larger | For server deployment |

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

| Config | Alpha | FPN | SimCC | Input | Params | Use Case |
|--------|-------|-----|-------|-------|--------|----------|
| **Mobile** | 0.35 | 32 | 96 | 224/256 | ~495K | Mobile, WASM, edge |
| **Server** | 1.0 | 48 | 128 | 320 | ~1.2M | Server, high accuracy |
| **Tiny** | 0.35 | 24 | 64 | 224 | ~105K | Ultra-constrained |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--alpha` | 0.35 | Backbone width multiplier |
| `--fpn_ch` | 32 | FPN channel dimension |
| `--simcc_ch` | 96 | SimCC head hidden channels |
| `--img_size` | 256 | Input image size |
| `--num_bins` | 256 | Number of classification bins (usually = img_size) |
| `--tau` | 1.0 | Softmax temperature |
| `--batch_size` | 512 | Training batch size |
| `--lr` | 0.001 | Initial learning rate |
| `--epochs` | 200 | Training epochs |

---

## Leaderboard

Evaluated on [DocCornerDataset](https://huggingface.co/datasets/mapo80/DocCornerDataset) validation split:

| Model | Input | mean_iou | Corner err (px) | Latency (ms) | Size |
|-------|-------|----------|-----------------|--------------|------|
| `mobilenetv2_224_best` | 224 | 0.9894 | 0.57 | 4.24 | 0.98 MB |
| `mobilenetv2_256_best` | 256 | **0.9902** | 0.60 | 8.18 | 0.98 MB |
| `mobilenetv2_320` | 320 | 0.9855 | 1.13 | 5.36 | 0.88 MB |
| `mobilenetv3_224` | 224 | 0.9842 | 0.86 | 3.96 | 1.47 MB |

**Recommended**: `mobilenetv2_224_best` - Best speed/accuracy/robustness tradeoff for deployment.

---

## Quick Start

### Install

```bash
git clone https://github.com/mapo80/DocCornerNet-CoordClass.git
cd DocCornerNet-CoordClass
pip install -r requirements.txt
```

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
├── model.py           # Network architecture
├── dataset.py         # Data loading + augmentation
├── losses.py          # Loss functions
├── metrics.py         # Evaluation metrics
├── train_ultra.py     # Training script (HF dataset support)
├── evaluate.py        # Evaluation
├── export.py          # Export (SavedModel, TFLite, ONNX)
├── export_onnx.py     # ONNX export
├── create_hf_dataset.py  # Create HuggingFace dataset
├── setup_remote.sh    # Remote machine setup
├── requirements.txt   # Dependencies
└── checkpoints/       # Pretrained models
    ├── mobilenetv2_224_best/
    ├── mobilenetv2_256_best/
    ├── mobilenetv2_320/
    └── mobilenetv3_224/
```

---

## References

- [SimCC](https://arxiv.org/abs/2107.03332) - ECCV 2022
- [MobileNetV3](https://arxiv.org/abs/1905.02244) - ICCV 2019
- [FPN](https://arxiv.org/abs/1612.03144) - CVPR 2017

## License

MIT License - see [LICENSE](LICENSE) file.
