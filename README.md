# DocCornerNet - Marginal Coordinate Classification

A lightweight neural network for document corner detection using **Marginal Coordinate Classification** (SimCC).

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-DocCornerDataset-yellow)](https://huggingface.co/datasets/mapo80/DocCornerDataset)

## Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/mapo80/DocCornerNet-CoordClass.git
cd DocCornerNet-CoordClass
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
pip install huggingface_hub
huggingface-cli download mapo80/DocCornerDataset --repo-type dataset --local-dir ./data
```

### 3. Train Teacher Model

```bash
python train.py \
    --data_root ./data \
    --batch_size 64 \
    --epochs 100 \
    --lr 0.0002 \
    --augment \
    --cache_images \
    --fast_mode \
    --output_dir ./checkpoints
```

### 4. Train Student Model (Optional - Faster Inference)

```bash
python train_student.py \
    --teacher_weights ./checkpoints/best_model.weights.h5 \
    --data_root ./data \
    --student_alpha 0.75 \
    --student_fpn_ch 32 \
    --student_simcc_ch 96 \
    --epochs 60 \
    --batch_size 64 \
    --augment \
    --cache_images \
    --fast_mode \
    --outlier_list ./data/outliers.txt \
    --output_dir ./checkpoints_student
```

### 5. Evaluate

```bash
# Teacher
python evaluate.py \
    --model_path ./checkpoints/best_model.weights.h5 \
    --data_root ./data \
    --split test

# Student
python evaluate.py \
    --model_path ./checkpoints_student/student_distill_*/best_student.weights.h5 \
    --data_root ./data \
    --split test \
    --fpn_ch 32 \
    --simcc_ch 96
```

### 6. Export to TFLite

```bash
# Teacher (float32)
python export_tflite.py \
    --model_path ./checkpoints/best_model.weights.h5 \
    --output ./exported_tflite/doccornernet_v3_224_float32.tflite

# Teacher (float16 - smaller)
python export_tflite.py \
    --model_path ./checkpoints/best_model.weights.h5 \
    --output ./exported_tflite/doccornernet_v3_224_float16.tflite \
    --float16

# Student (float16)
python export_tflite.py \
    --model_path ./checkpoints_student/student_distill_*/best_student.weights.h5 \
    --output ./exported_tflite/doccornernet_v3_student_224_float16.tflite \
    --float16
```

**TFLite Output Format:** `[1, 9]` = `[x0, y0, x1, y1, x2, y2, x3, y3, score]`

---

## Results

### Expected Performance (after training)

| Model | Parameters | Mean IoU | Corner Error | R@90 | R@95 | TFLite (ms) | Size |
|-------|------------|----------|--------------|------|------|-------------|------|
| **Teacher** | 742,417 | ~98% | ~1.0 px | ~99% | ~97% | ~5 ms | 1.47 MB |
| **Student** | 669,761 | ~97% | ~1.3 px | ~98% | ~95% | ~3.5 ms | 1.34 MB |

*TFLite float16, Apple M2 Pro CPU, batch size 1.*

### Key Insights

- **Teacher**: Best accuracy (~1px corner error, ~97% R@95)
- **Student**: 1.5x faster, 10% fewer params, trades ~1% IoU

> **Note**: The pretrained checkpoints in this repo require retraining due to model architecture updates. Use the training commands above to train fresh models.

---

## Architecture

```
Input [224×224×3]
       ↓
MobileNetV3-Small (α=0.75)
       ↓
Mini-FPN (48 ch) → Feature Map [56×56×48]
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
3. Spatial awareness preserved (FPN + 1D conv vs GAP)
4. Sub-pixel precision via soft-argmax

---

## Training Parameters

| Parameter | Teacher | Student |
|-----------|---------|---------|
| `--alpha` | 0.75 | 0.75 |
| `--fpn_ch` | 48 | 32 |
| `--simcc_ch` | 128 | 96 |
| `--batch_size` | 64 | 64 |
| `--epochs` | 100 | 60 |
| `--lr` | 0.0002 | 0.0002 |

### Model Presets

| Preset | Flags | Params |
|--------|-------|--------|
| Teacher | `--alpha 0.75 --fpn_ch 48 --simcc_ch 128` | 742,417 |
| Student | `--alpha 0.75 --fpn_ch 32 --simcc_ch 96` | 669,761 |
| Lite | `--alpha 0.5 --fpn_ch 32 --simcc_ch 96` | 360,121 |
| Tiny | `--alpha 0.35 --fpn_ch 24 --simcc_ch 64 --backbone_minimalistic` | 104,513 |

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
├── checkpoints/                    # Teacher model
│   ├── best_model.weights.h5
│   └── config.json
├── checkpoints_student/            # Student model
│   └── student_distill_*/
├── exported_tflite/                # TFLite exports
│   ├── doccornernet_v3_224_float16.tflite
│   └── doccornernet_v3_student_224_float16.tflite
├── model.py                        # Network architecture
├── dataset.py                      # Dataset with augmentations
├── train.py                        # Teacher training
├── train_student.py                # Student distillation
├── evaluate.py                     # Evaluation
├── export_tflite.py                # TFLite export
└── README.md
```

---

## References

- [SimCC](https://arxiv.org/abs/2107.03332) - ECCV 2022
- [MobileNetV3](https://arxiv.org/abs/1905.02244) - ICCV 2019
- [FPN](https://arxiv.org/abs/1612.03144) - CVPR 2017

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
