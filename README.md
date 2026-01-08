# DocCornerNet - Marginal Coordinate Classification

A lightweight neural network for document corner detection using **Marginal Coordinate Classification** (SimCC).

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-DocCornerDataset-yellow)](https://huggingface.co/datasets/mapo80/DocCornerDataset)

## Leaderboard

| Model | Img | mean_iou | Corner err (px) | Latency (ms) | Size |
|-------|-----|----------|-----------------|--------------|------|
| `mobilenetv2_224_best` | 224 | 0.9894 | 0.57 | 4.24 | 0.98 MB |
| `mobilenetv2_256_best` | 256 | **0.9902** | 0.60 | 8.18 | 0.98 MB |
| `mobilenetv2_320` | 320 | 0.9855 | 1.13 | 5.36 | 0.88 MB |
| `mobilenetv3_224` | 224 | 0.9842 | 0.86 | 3.96 | 1.47 MB |

**Winner**: `mobilenetv2_224_best` - Best tradeoff for deployment (smallest model, fastest, most robust).

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
