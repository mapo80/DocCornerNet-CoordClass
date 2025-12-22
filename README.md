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
    --split test \
    --input_norm imagenet

# Student
python evaluate.py \
    --model_path ./checkpoints_student/student_distill_*/best_student.weights.h5 \
    --data_root ./data \
    --split test \
    --input_norm imagenet \
    --fpn_ch 32 \
    --simcc_ch 96

# MobileNetV2 small
python evaluate.py \
    --model_path ./checkpoints/v3_mnv2_small_aug_outliers_20251220_111218 \
    --data_root ./data \
    --split test
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

# MobileNetV2 small (float16)
python export_tflite.py \
    --model_path ./checkpoints/v3_mnv2_small_aug_outliers_20251220_111218/best_model.weights.h5 \
    --output ./exported_tflite/doccornernet_v3_mnv2_small_float16.tflite \
    --float16

# MobileNetV2 small @256 (WASM-friendly input: raw float32 in [0..255], normalization inside the model)
python export_tflite.py \
    --model_path ./checkpoints/v3_mnv2_small_256_aug_outliers_*/best_model.weights.h5 \
    --output ./exported_tflite/doccornernet_v3_mnv2_small_256_raw255_float32.tflite \
    --tflite_input_norm raw255
```

**TFLite Output Format:** `[1, 9]` = `[x0, y0, x1, y1, x2, y2, x3, y3, score]`

### 7. Evaluate / Benchmark on TFLite

```bash
# Full metrics (mirrors evaluate.py, but runs inference via TFLite Interpreter)
python eval_tflite.py \
  --tflite_models ./exported_tflite/doccornernet_v3_mnv2_small_256_float32.tflite \
  --data_root ./data \
  --split val \
  --input_norm imagenet \
  --threads 1

# If you exported a *_raw255_*.tflite model (preprocessing inside the model)
python eval_tflite.py \
  --tflite_models ./exported_tflite/doccornernet_v3_mnv2_small_256_raw255_float32.tflite \
  --data_root ./data \
  --split val \
  --input_norm raw255 \
  --threads 1

# Microbenchmark (latency only, no image decoding/preprocess)
python benchmark_tflite.py \
  --model ./exported_tflite/doccornernet_v3_mnv2_small_256_float32.tflite \
  --threads 1
```

---

## Results

### Best Checkpoints (DocScannerDetection dataset)

These checkpoints are trained/evaluated on `doc-scanner-dataset-labeled` (with `train_cleaned_plus_outliers` / `val_cleaned` splits) and include outlier handling.

| Checkpoint | Backbone | Img | Params | val_cleaned mIoU | val mIoU | val_removed mIoU | val_outliers mIoU |
|-----------|----------|-----|--------|------------------|---------|------------------|-------------------|
| `checkpoints/mobilenetv2_224_best` | MNv2 α=0.35 | 224 | 495,353 | 0.9894 | **0.9826** | **0.9429** | **0.7075** |
| `checkpoints/mobilenetv2_256_best` | MNv2 α=0.35 | 256 | 495,353 | **0.9902** | 0.9819 | 0.9341 | 0.6281 |
| `checkpoints/mobilenetv3_224` | MNv3-S α=0.75 | 224 | 742,417 | 0.9842 | 0.9734 | 0.9107 | 0.5960 |
| `checkpoints/mobilenetv2_320` | MNv2 α=0.35 | 320 | 444,841 | 0.9855 | 0.9757 | 0.9190 | 0.5645 |

**Winner (overall): `checkpoints/mobilenetv2_224_best`**
- Best tradeoff for deployment: strongest robustness (`val_removed` / `val_outliers`) with the smallest model and 224px input.
- If you must hit **Mean IoU ≥ 0.99 on `val_cleaned`**, pick `checkpoints/mobilenetv2_256_best` (slower due to 256px).

### Previous Checkpoints (same dataset, for reference)

| Checkpoint | Img | Params | val_cleaned mIoU | val mIoU | val_removed mIoU | val_outliers mIoU |
|-----------|-----|--------|------------------|---------|------------------|-------------------|
| `checkpoints/v3_mnv2_small_aug_outliers_20251220_111218` | 224 | 495,353 | 0.9876 | 0.9805 | 0.9398 | 0.6934 |
| `checkpoints/v3_mnv2_small_256_aug_outliers_20251221_100332` | 256 | 495,353 | 0.9900 | 0.9822 | 0.9369 | 0.6856 |

### Legacy (different dataset)

- `checkpoints/best_model.weights.h5` and `checkpoints_student/student_distill_20251219_153437` are pretrained on the HuggingFace `DocCornerDataset` and are **not directly comparable** to the tables above (they perform poorly on `doc-scanner-dataset-labeled` without re-training).
- PyTorch regression baselines live in `/Volumes/ZX20/ML-Models/DocCornerNet-Regression/checkpoints/` (e.g. `best_224.pth`, `best_320.pth`) and are trained/evaluated on `DocCornerDataset` (reported ≈95–96% mIoU in that repo’s `README.md`).

### TFLite Benchmarks (measured)

Latency/size are mostly architecture-dependent (weights and dataset don’t materially change them).

| Model (architecture) | Parameters | TFLite (ms) | Size |
|-------|------------|-------------|------|
| **MNv3-S α=0.75 (teacher preset)** | 742,417 | 4.93 ms | 1.47 MB |
| **Student (distill preset)** | 669,761 | 3.38 ms | 1.34 MB |
| **MobileNetV2 small (α=0.35) @224** | 495,353 | 3.78 ms | 0.98 MB |
| **MobileNetV2 small (α=0.35) @256** | 495,353 | 4.81 ms | 0.98 MB |

Notes:
- Latency: `benchmark_tflite.py` p50, CPU (XNNPACK), batch size 1.
- Size: `.tflite` float16 file size on disk.

### Key Insights

- For `doc-scanner-dataset-labeled`, prefer **`checkpoints/mobilenetv2_224_best`** unless you explicitly need 256px.
- 256px gives slightly higher `val_cleaned` mIoU, but it’s slower (more pixels) and is less robust on outliers in our current best checkpoint.
- `checkpoints/mobilenetv3_224` is heavier and underperforms MNv2 on this dataset as trained so far.

> Note: `checkpoints/best_model.keras` may fail to load on Keras 3 due to legacy `Lambda` layers; use `checkpoints/best_model.weights.h5` for evaluation/export.

### 256px Recipe (targeting ~99% Mean IoU)

The final deployment target (WASM/TFLite) benefits from a small backbone. A practical path is:
1) fine-tune a strong teacher at 256
2) distill into **MobileNetV2 small** at 256
3) mine hard samples and fine-tune

```bash
# 1) Teacher @256 (warm-start from the 224 teacher)
python train.py \
  --data_root ./data \
  --img_size 256 --num_bins 256 \
  --backbone mobilenetv3_small --alpha 0.75 --fpn_ch 48 --simcc_ch 128 \
  --init_weights ./checkpoints/best_model.weights.h5 \
  --batch_size 64 --epochs 30 --lr 0.0001 \
  --augment --cache_images --fast_mode \
  --outlier_list ./data/outliers.txt --outlier_weight 3.0 \
  --output_dir ./checkpoints --experiment_name v3_teacher_256_ft

# 2) Student @256 (MobileNetV2 small, warm-start from the 224 MNv2 small)
python train_student.py \
  --teacher_weights ./checkpoints/v3_teacher_256_ft_*/best_model.weights.h5 \
  --data_root ./data \
  --img_size 256 --num_bins 256 \
  --student_backbone mobilenetv2 --student_alpha 0.35 --student_fpn_ch 32 --student_simcc_ch 96 \
  --student_init_weights ./checkpoints/v3_mnv2_small_aug_outliers_*/best_model.weights.h5 \
  --batch_size 64 --epochs 60 --lr 0.0002 \
  --augment --cache_images --fast_mode \
  --outlier_list ./data/outliers.txt --outlier_weight 3.0 \
  --output_dir ./checkpoints_student --experiment_name student_mnv2_256_distill

# 3) Mine hard positives (low IoU) on train and fine-tune with higher outlier_weight
python mine_outliers.py \
  --model_path ./checkpoints_student/student_mnv2_256_distill_*/best_student.weights.h5 \
  --data_root ./data \
  --split train \
  --batch_size 64 \
  --input_norm imagenet \
  --iou_threshold 0.98 \
  --include_fn \
  --output ./data/outliers_mined_256_iou98.txt \
  --cache_images --fast_mode

python train_student.py \
  --teacher_weights ./checkpoints/v3_teacher_256_ft_*/best_model.weights.h5 \
  --data_root ./data \
  --img_size 256 --num_bins 256 \
  --student_backbone mobilenetv2 --student_alpha 0.35 --student_fpn_ch 32 --student_simcc_ch 96 \
  --student_init_weights ./checkpoints_student/student_mnv2_256_distill_*/best_student.weights.h5 \
  --batch_size 64 --epochs 20 --lr 0.00005 \
  --augment --cache_images --fast_mode \
  --outlier_list ./data/outliers_mined_256_iou98.txt --outlier_weight 5.0 \
  --output_dir ./checkpoints_student --experiment_name student_mnv2_256_ft_mined
```

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
├── eval_tflite.py                  # Evaluation via TFLite Interpreter
├── benchmark_tflite.py             # TFLite microbenchmark (CPU)
├── mine_outliers.py                # Hard-sample mining (outlier list)
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
