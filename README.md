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

### Configuration Reference (`train_ultra.py`)

> **Note:** `evaluate.py` and `export.py` use different defaults (alpha=0.75, fpn_ch=48, simcc_ch=128, backbone=mobilenetv3_small) targeting the "server" config. In practice, both scripts auto-load `config.json` from checkpoint directories, so these defaults are rarely needed.

#### Data Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data_root` | str | None | Root directory with `images/`, `labels/`, split files. Mutually exclusive with `--hf_dataset` |
| `--hf_dataset` | str | None | HuggingFace dataset name (e.g. `mapo80/DocCornerDataset`) or path to local Parquet directory |
| `--download_hf` | str | None | Download HuggingFace dataset to this directory and exit (use with `--hf_dataset`) |
| `--hf_token` | str | None | HuggingFace API token for private datasets (or set `HF_TOKEN` env var) |
| `--output_dir` | str | None | Output directory for checkpoints (required for training) |
| `--train_split` | str | `"train"` | Training split name |
| `--val_split` | str | `"validation"` | Validation split name. Auto-corrected to `"val"` when using `--data_root` (local datasets) |
| `--experiment_name` | str | None | Custom experiment subdirectory name under `--output_dir` |

#### Model Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--backbone` | str | `"mobilenetv2"` | Backbone architecture: `mobilenetv2`, `mobilenetv3_small`, `mobilenetv3_large` |
| `--alpha` | float | 0.35 | Backbone width multiplier (0.35 = mobile, 1.0 = server) |
| `--backbone_weights` | str | `"imagenet"` | Backbone init weights (`"imagenet"` or `None` to skip download) |
| `--init_weights` | str | None | Warm-start from existing weights (`.weights.h5` file or checkpoint dir). Useful for fine-tuning at different `img_size`/`num_bins` |
| `--init_partial` | flag | False | If strict weight loading fails, retry with `by_name=True, skip_mismatch=True` (HDF5 only) |
| `--fpn_ch` | int | 32 | FPN channel dimension |
| `--simcc_ch` | int | 96 | SimCC head hidden channels |
| `--img_size` | int | 256 | Input image size (square, in pixels) |
| `--num_bins` | int | 256 | Number of classification bins per axis (usually = `img_size`) |
| `--tau` | float | 1.0 | Softmax temperature for SimCC decoding. Lower = sharper distributions |

#### Loss Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sigma_px` | float | 3.0 | Gaussian sigma (in pixels) for SimCC 1D target distributions |
| `--w_simcc` | float | 1.0 | Weight for SimCC cross-entropy classification loss (applied to positive samples only) |
| `--w_coord` | float | 0.5 | Weight for coordinate L1 auxiliary loss (applied to positive samples only) |
| `--w_score` | float | 0.5 | Weight for document presence binary cross-entropy loss (applied to all samples) |
| `--label_smoothing` | float | 0.0 | Label smoothing for SimCC target distributions (0.0 = disabled) |
| `--ema_decay` | float | 0.0 | Exponential Moving Average decay rate (0.0 = disabled, 0.999 = typical) |

#### Training Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--batch_size` | int | 128 | Training batch size |
| `--epochs` | int | 100 | Number of training epochs |
| `--lr` | float | 3e-4 | Initial learning rate |
| `--weight_decay` | float | 1e-4 | L2 weight decay (AdamW) |
| `--warmup_epochs` | int | 5 | Linear learning rate warmup epochs |
| `--patience` | int | 20 | Early stopping patience (epochs without val improvement) |
| `--lr_patience` | int | 7 | ReduceLROnPlateau patience (epochs without val improvement) |
| `--lr_factor` | float | 0.5 | Learning rate reduction factor when plateau is detected |
| `--min_lr` | float | 1e-6 | Minimum learning rate floor |
| `--lr_schedule` | str | `"plateau"` | LR schedule: `"plateau"` (ReduceLROnPlateau) or `"cosine"` (cosine annealing) |

#### Data Loading & Augmentation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--num_workers` | int | 64 | Number of threads for parallel image loading |
| `--augment` | flag | False | Enable data augmentation during training |
| `--rotation_range` | float | 0.0 | Random rotation range in degrees (requires `--augment`; auto-disabled if `ImageProjectiveTransformV3` is unavailable) |

When `--augment` is enabled, the following augmentation config is applied (defined in `dataset.py`):

| Augmentation | Default Value | Description |
|-------------|---------------|-------------|
| `rotation_degrees` | 5 | Random rotation range (degrees) |
| `scale_range` | (0.9, 1.0) | Min/max scale factors |
| `brightness` | 0.2 | Brightness jitter range |
| `contrast` | 0.2 | Contrast jitter range |
| `saturation` | 0.1 | Saturation jitter range |
| `blur_prob` | 0.1 | Probability of Gaussian blur |
| `blur_kernel` | 3 | Gaussian blur kernel size |
| `translate` | 0.0 | Translation augmentation (disabled) |
| `perspective` | (0.0, 0.03) | Min/max perspective transform coefficients |

#### Environment Variables

| Variable | Set By | Description |
|----------|--------|-------------|
| `HF_TOKEN` / `HUGGINGFACE_TOKEN` | User | HuggingFace API token (alternative to `--hf_token`) |
| `TF_XLA_FLAGS` | Auto | Set to `--tf_xla_auto_jit=2` when NVIDIA GPU is detected |
| `TF_GPU_THREAD_MODE` | Auto | Set to `gpu_private` when NVIDIA GPU is detected |

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
    --weights ./checkpoints/mobilenetv2_256_best/best_model.weights.h5 \
    --output_dir ./exported \
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
├── model.py              # Network architecture (backbone, FPN, SimCC head, score head)
├── dataset.py            # Data loading, augmentation, HF/local dataset support
├── losses.py             # Loss functions (SimCC, Coord, Score) + training wrapper
├── metrics.py            # Evaluation metrics (polygon IoU, corner error, recall)
├── train_ultra.py        # Training script (HF dataset support, multi-platform)
├── evaluate.py           # Evaluation pipeline with detailed metric reporting
├── export.py             # Export to SavedModel, TFLite (float/int8), ONNX
├── export_onnx.py        # Dedicated ONNX export helper
├── create_hf_dataset.py  # Create HuggingFace Parquet datasets from local data
├── __init__.py            # Package initialization
├── setup_remote.sh       # Remote machine setup (RunPod, Lambda Labs, etc.)
├── sync_to_runpod.sh     # Sync files to RunPod instance
├── sync_to_vastai.sh     # Sync files to Vast.ai instance
├── requirements.txt      # Python dependencies
├── MODEL.md              # Detailed technical architecture documentation
└── checkpoints/          # Pretrained models
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
