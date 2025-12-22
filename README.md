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

# MobileNetV2 @256 (PTQ quantization)
# - int8 (hybrid): faster but may reduce accuracy
python export_tflite_int8.py \
    --checkpoint ./checkpoints/mobilenetv2_256_best \
    --data_root /path/to/doc-scanner-dataset-labeled \
    --split val_cleaned \
    --quantization int8 \
    --allow_float_fallback \
    --output ./exported_tflite/doccornernet_v3_mnv2_256_best_int8_hybrid.tflite

# - dynamic range (weights-only): matches float16/float32 accuracy, but can reduce XNNPACK coverage
python export_tflite_int8.py \
    --checkpoint ./checkpoints/mobilenetv2_256_best \
    --data_root /path/to/doc-scanner-dataset-labeled \
    --split val_cleaned \
    --quantization dynamic \
    --output ./exported_tflite/doccornernet_v3_mnv2_256_best_dynamic.tflite
```

#### Keras (.weights.h5 / checkpoint dir) → TFLite (float32 / float16 / int8 full / int8 dynamic)

Notes:
- Prefer exporting from `best_model.weights.h5` (or a checkpoint dir containing it). Loading `*.keras` may fail on some Keras 3 setups (legacy `Lambda`).
- Always pass `--config .../config.json` when exporting from a checkpoint directory to avoid picking the wrong config.
- Dataset path (`DATA_ROOT`) is required for **int8 full** PTQ calibration (and currently required by the script even for `dynamic`).

Set your dataset root:

```bash
export DATA_ROOT=/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-labeled
```

**float32** (coords9 output `[1,9]`):

```bash
python export_tflite.py --model_path ./checkpoints/mobilenetv2_224_best --config ./checkpoints/mobilenetv2_224_best/config.json --output ./exported_tflite/doccornernet_v3_mnv2_224_best_float32.tflite
python export_tflite.py --model_path ./checkpoints/mobilenetv2_256_best --config ./checkpoints/mobilenetv2_256_best/config.json --output ./exported_tflite/doccornernet_v3_mnv2_256_best_float32.tflite
python export_tflite.py --model_path ./checkpoints/mobilenetv2_320      --config ./checkpoints/mobilenetv2_320/config.json      --output ./exported_tflite/doccornernet_v3_mnv2_320_float32.tflite
python export_tflite.py --model_path ./checkpoints/mobilenetv3_224      --config ./checkpoints/mobilenetv3_224/config.json      --output ./exported_tflite/doccornernet_v3_mnv3_224_float32.tflite
```

**float16** (coords9 output `[1,9]`, smaller):

```bash
python export_tflite.py --model_path ./checkpoints/mobilenetv2_224_best --config ./checkpoints/mobilenetv2_224_best/config.json --output ./exported_tflite/doccornernet_v3_mnv2_224_best_float16_xnnpack_full_nearestmul.tflite --float16
python export_tflite.py --model_path ./checkpoints/mobilenetv2_256_best --config ./checkpoints/mobilenetv2_256_best/config.json --output ./exported_tflite/doccornernet_v3_mnv2_256_best_float16_xnnpack_full_nearestmul.tflite --float16
python export_tflite.py --model_path ./checkpoints/mobilenetv2_320      --config ./checkpoints/mobilenetv2_320/config.json      --output ./exported_tflite/doccornernet_v3_mnv2_320_float16_xnnpack_full_nearestmul.tflite      --float16
python export_tflite.py --model_path ./checkpoints/mobilenetv3_224      --config ./checkpoints/mobilenetv3_224/config.json      --output ./exported_tflite/doccornernet_v3_mnv3_224_float16_xnnpack_full_nearestmul.tflite      --float16
```

**int8 full quant** (I/O int8, outputs packed SimCC logits + score logit; decode outside the model):

```bash
python export_tflite_int8.py --checkpoint ./checkpoints/mobilenetv2_224_best --data_root "$DATA_ROOT" --split val_cleaned --num_calib 500 --quantization int8 --io_dtype int8 --output_dtype int8 --output_format simcc_logits --simcc_packed_layout bins_first --axis_mean_impl dwconv_full --global_pool_impl dwconv_strided --output ./exported_tflite/doccornernet_v3_mnv2_224_best_int8_full_simcc_int8io_xnnpackfull_stridedgap_binsfirst.tflite --threads 4
python export_tflite_int8.py --checkpoint ./checkpoints/mobilenetv2_256_best --data_root "$DATA_ROOT" --split val_cleaned --num_calib 500 --quantization int8 --io_dtype int8 --output_dtype int8 --output_format simcc_logits --simcc_packed_layout bins_first --axis_mean_impl dwconv_full --global_pool_impl dwconv_strided --output ./exported_tflite/doccornernet_v3_mnv2_256_best_int8_full_simcc_int8io_xnnpackfull_stridedgap_binsfirst.tflite --threads 4
python export_tflite_int8.py --checkpoint ./checkpoints/mobilenetv2_320      --data_root "$DATA_ROOT" --split val_cleaned --num_calib 500 --quantization int8 --io_dtype int8 --output_dtype int8 --output_format simcc_logits --simcc_packed_layout bins_first --axis_mean_impl dwconv_full --global_pool_impl dwconv_strided --output ./exported_tflite/doccornernet_v3_mnv2_320_int8_full_simcc_int8io_xnnpackfull_stridedgap_binsfirst.tflite      --threads 4
python export_tflite_int8.py --checkpoint ./checkpoints/mobilenetv3_224      --data_root "$DATA_ROOT" --split val_cleaned --num_calib 500 --quantization int8 --io_dtype int8 --output_dtype int8 --output_format simcc_logits --simcc_packed_layout bins_first --axis_mean_impl dwconv_full --global_pool_impl dwconv_strided --output ./exported_tflite/doccornernet_v3_mnv3_224_int8_full_simcc_int8io_xnnpackfull_stridedgap_binsfirst.tflite      --threads 4
```

**int8 dynamic range** (weights-only; float32 I/O; coords9 output `[1,9]`):

```bash
python export_tflite_int8.py --checkpoint ./checkpoints/mobilenetv2_224_best --data_root "$DATA_ROOT" --split val_cleaned --quantization dynamic --output ./exported_tflite/doccornernet_v3_mnv2_224_best_dynamic.tflite --threads 4
python export_tflite_int8.py --checkpoint ./checkpoints/mobilenetv2_256_best --data_root "$DATA_ROOT" --split val_cleaned --quantization dynamic --output ./exported_tflite/doccornernet_v3_mnv2_256_best_dynamic.tflite --threads 4
python export_tflite_int8.py --checkpoint ./checkpoints/mobilenetv2_320      --data_root "$DATA_ROOT" --split val_cleaned --quantization dynamic --output ./exported_tflite/doccornernet_v3_mnv2_320_dynamic.tflite      --threads 4
python export_tflite_int8.py --checkpoint ./checkpoints/mobilenetv3_224      --data_root "$DATA_ROOT" --split val_cleaned --quantization dynamic --output ./exported_tflite/doccornernet_v3_mnv3_224_dynamic.tflite      --threads 4
```

#### WASM / XNNPACK full delegation (verification)

For the WebAssembly runtime (`document-scanner-wasm`), we want **100% of the TFLite graph delegated to XNNPACK** (i.e., the post-delegate execution plan contains only `DELEGATE` nodes). This required removing a few ops that XNNPACK does not delegate in practice (e.g. `STRIDED_SLICE`, `TILE`, `PACK`, `SUM`, `RESIZE_NEAREST_NEIGHBOR`).

What changed in this repo (`model.py`) to enable full delegation:
- `Resize1D`: avoids `STRIDED_SLICE` by reshaping away the singleton dimension after resize.
- `Broadcast1D`: avoids `TILE` by using `MUL` broadcasting with a constant ones tensor.
- `SimCCDecode`: avoids `SUM`/`PACK`/`STRIDED_SLICE` by computing the expectation via `matmul` and reshaping/concatenating.
- `NearestUpsample2x`: implements 2× nearest upsampling via `RESHAPE+MUL` (no `RESIZE_NEAREST_NEIGHBOR`, no `TILE`).

Accuracy sanity check (DocScannerDetection `val_cleaned`): the fully-delegated float16 exports match the baseline float16 mIoU for MNv2@224_best (0.9894), MNv2@256_best (0.9902), MNv2@320 (0.9855), MNv3@224 (0.9842). See `exported_tflite/eval_float16_full_delegate_improve.json` and `exported_tflite/eval_xnnpack_full_delegate_others.json`.

Export the fully-delegated float16 models (always pass `--config` to avoid picking `./checkpoints/config.json` by mistake):

```bash
# MobileNetV2 @224 (best)
python export_tflite.py \
  --model_path ./checkpoints/mobilenetv2_224_best \
  --config ./checkpoints/mobilenetv2_224_best/config.json \
  --output ./exported_tflite/doccornernet_v3_mnv2_224_best_float16_xnnpack_full_nearestmul.tflite \
  --float16

# MobileNetV2 @256 (best)
python export_tflite.py \
  --model_path ./checkpoints/mobilenetv2_256_best \
  --config ./checkpoints/mobilenetv2_256_best/config.json \
  --output ./exported_tflite/doccornernet_v3_mnv2_256_best_float16_xnnpack_full_nearestmul.tflite \
  --float16

# MobileNetV2 @320
python export_tflite.py \
  --model_path ./checkpoints/mobilenetv2_320 \
  --config ./checkpoints/mobilenetv2_320/config.json \
  --output ./exported_tflite/doccornernet_v3_mnv2_320_float16_xnnpack_full_nearestmul.tflite \
  --float16

# MobileNetV3-Small @224
python export_tflite.py \
  --model_path ./checkpoints/mobilenetv3_224 \
  --config ./checkpoints/mobilenetv3_224/config.json \
  --output ./exported_tflite/doccornernet_v3_mnv3_224_float16_xnnpack_full_nearestmul.tflite \
  --float16
```

Verify XNNPACK coverage using the helper in the WASM repo:

```bash
cd /Volumes/ZX20/ML-Models/document-scanner/document-scanner-wasm
cmake -S tests -B build-native
cmake --build build-native --target xnnpack_delegate_report -j 8

./build-native/xnnpack_delegate_report \
  /Volumes/ZX20/ML-Models/DocCornerNet-CoordClass/exported_tflite/doccornernet_v3_mnv2_256_best_float16_xnnpack_full_nearestmul.tflite \
  4

# Expected: "Non-delegated builtin ops (plan):" is empty.
# (Execution plan nodes may be 1 or more; when fully delegated, all plan nodes are DELEGATE.)
```

You can use the same check for **any** model (float32/float16/int8/dynamic). Example (INT8 full):

```bash
./build-native/xnnpack_delegate_report \
  /Volumes/ZX20/ML-Models/DocCornerNet-CoordClass/exported_tflite/doccornernet_v3_mnv2_224_best_int8_full_simcc_int8io_xnnpackfull_stridedgap_binsfirst.tflite \
  4
```

Batch-check multiple exports (fail-fast if any model is not fully delegated):

```bash
MODELS=(
  /Volumes/ZX20/ML-Models/DocCornerNet-CoordClass/exported_tflite/doccornernet_v3_mnv2_224_best_float16_xnnpack_full_nearestmul.tflite
  /Volumes/ZX20/ML-Models/DocCornerNet-CoordClass/exported_tflite/doccornernet_v3_mnv2_224_best_int8_full_simcc_int8io_xnnpackfull_stridedgap_binsfirst.tflite
  /Volumes/ZX20/ML-Models/DocCornerNet-CoordClass/exported_tflite/doccornernet_v3_mnv2_256_best_float16_xnnpack_full_nearestmul.tflite
  /Volumes/ZX20/ML-Models/DocCornerNet-CoordClass/exported_tflite/doccornernet_v3_mnv2_256_best_int8_full_simcc_int8io_xnnpackfull_stridedgap_binsfirst.tflite
  /Volumes/ZX20/ML-Models/DocCornerNet-CoordClass/exported_tflite/doccornernet_v3_mnv2_320_float16_xnnpack_full_nearestmul.tflite
  /Volumes/ZX20/ML-Models/DocCornerNet-CoordClass/exported_tflite/doccornernet_v3_mnv2_320_int8_full_simcc_int8io_xnnpackfull_stridedgap_binsfirst.tflite
  /Volumes/ZX20/ML-Models/DocCornerNet-CoordClass/exported_tflite/doccornernet_v3_mnv3_224_float16_xnnpack_full_nearestmul.tflite
  /Volumes/ZX20/ML-Models/DocCornerNet-CoordClass/exported_tflite/doccornernet_v3_mnv3_224_int8_full_simcc_int8io_xnnpackfull_stridedgap_binsfirst.tflite
)

for m in "${MODELS[@]}"; do
  echo "==> $m"
  out=$(./build-native/xnnpack_delegate_report "$m" 4)
  if echo "$out" | grep -qE '^  '; then
    echo "$out"
    echo "NOT FULLY DELEGATED"
    exit 1
  fi
done

echo "OK: all models fully delegated to XNNPACK."
```

Optional: update the WASM model files (used by the package) with the fully-delegated exports:

```bash
cp -f /Volumes/ZX20/ML-Models/DocCornerNet-CoordClass/exported_tflite/doccornernet_v3_mnv2_224_best_float16_xnnpack_full_nearestmul.tflite \
  /Volumes/ZX20/ML-Models/document-scanner/document-scanner-wasm/models/mnv2_224_best_float16.tflite
cp -f /Volumes/ZX20/ML-Models/DocCornerNet-CoordClass/exported_tflite/doccornernet_v3_mnv2_256_best_float16_xnnpack_full_nearestmul.tflite \
  /Volumes/ZX20/ML-Models/document-scanner/document-scanner-wasm/models/mnv2_256_best_float16.tflite
cp -f /Volumes/ZX20/ML-Models/DocCornerNet-CoordClass/exported_tflite/doccornernet_v3_mnv2_320_float16_xnnpack_full_nearestmul.tflite \
  /Volumes/ZX20/ML-Models/document-scanner/document-scanner-wasm/models/mnv2_320_float16.tflite
cp -f /Volumes/ZX20/ML-Models/DocCornerNet-CoordClass/exported_tflite/doccornernet_v3_mnv3_224_float16_xnnpack_full_nearestmul.tflite \
  /Volumes/ZX20/ML-Models/document-scanner/document-scanner-wasm/models/mnv3_224_float16.tflite
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

### TFLite Float16 Leaderboard (measured, full-delegate)

All models below are **float16 exports** (float32 input) and **full-delegate** (post-delegate execution plan contains only `DELEGATE` nodes).

Evaluation command (metrics + latency, `Invoke()` only):

```bash
python eval_tflite.py \
  --tflite_models \
    exported_tflite/doccornernet_v3_mnv2_224_best_float16_xnnpack_full_nearestmul.tflite \
    exported_tflite/doccornernet_v3_mnv2_256_best_float16_xnnpack_full_nearestmul.tflite \
    exported_tflite/doccornernet_v3_mnv2_320_float16_xnnpack_full_nearestmul.tflite \
    exported_tflite/doccornernet_v3_mnv3_224_float16_xnnpack_full_nearestmul.tflite \
  --data_root /Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-labeled \
  --split val_cleaned \
  --input_norm imagenet \
  --threads 4 \
  --benchmark_runs 200 \
  --output exported_tflite/eval_float16_full_leaderboard_coords9.json
```

Leaderboard:

| Model | Img | mean_iou | Corner err (mean / p95) | Recall@95 | Latency p50 / p95 (ms) | Size |
|------|-----|----------|--------------------------|-----------|-------------------------|------|
| `exported_tflite/doccornernet_v3_mnv2_224_best_float16_xnnpack_full_nearestmul.tflite` | 224 | 0.9894 | 0.57 / 1.44 px | 99.8% | 4.24 / 6.76 | 0.978 MB |
| `exported_tflite/doccornernet_v3_mnv2_256_best_float16_xnnpack_full_nearestmul.tflite` | 256 | **0.9902** | 0.60 / 1.52 px | 100.0% | 8.18 / 16.53 | 0.978 MB |
| `exported_tflite/doccornernet_v3_mnv2_320_float16_xnnpack_full_nearestmul.tflite` | 320 | 0.9855 | 1.13 / 2.57 px | 98.4% | 5.36 / 8.14 | 0.883 MB |
| `exported_tflite/doccornernet_v3_mnv3_224_float16_xnnpack_full_nearestmul.tflite` | 224 | 0.9842 | 0.86 / 2.22 px | 98.0% | 3.96 / 7.06 | 1.471 MB |

### TFLite INT8 Leaderboard (measured, SimCC logits decode outside)

All models below are **INT8 full-delegate** (post-delegate execution plan contains only `DELEGATE` nodes).

Evaluation command (metrics + latency, `Invoke()` only):

```bash
python eval_tflite_simcc.py \
  --tflite_models \
    exported_tflite/doccornernet_v3_mnv2_224_best_int8_full_simcc_int8io_xnnpackfull_stridedgap_binsfirst.tflite \
    exported_tflite/doccornernet_v3_mnv2_256_best_int8_full_simcc_int8io_xnnpackfull_stridedgap_binsfirst.tflite \
    exported_tflite/doccornernet_v3_mnv2_320_int8_full_simcc_int8io_xnnpackfull_stridedgap_binsfirst.tflite \
    exported_tflite/doccornernet_v3_mnv3_224_int8_full_simcc_int8io_xnnpackfull_stridedgap_binsfirst.tflite \
  --data_root /Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-labeled \
  --split val_cleaned \
  --input_norm imagenet \
  --tau 1.0 \
  --threads 4 \
  --benchmark_runs 200 \
  --output exported_tflite/eval_int8_full_leaderboard_simcc_binsfirst.json
```

Leaderboard (higher `mean_iou` is better; lower latency is better):

| Model | Img | mean_iou | Corner err (mean / p95) | Recall@95 | Latency p50 / p95 (ms) | Size |
|------|-----|----------|--------------------------|-----------|-------------------------|------|
| `exported_tflite/doccornernet_v3_mnv2_224_best_int8_full_simcc_int8io_xnnpackfull_stridedgap_binsfirst.tflite` | 224 | **0.9888** | **0.59 / 1.52 px** | **99.9%** | **2.53 / 4.50** | 0.824 MB |
| `exported_tflite/doccornernet_v3_mnv2_256_best_int8_full_simcc_int8io_xnnpackfull_stridedgap_binsfirst.tflite` | 256 | **0.9893** | 0.65 / 1.64 px | 99.8% | 2.92 / 5.20 | 0.839 MB |
| `exported_tflite/doccornernet_v3_mnv2_320_int8_full_simcc_int8io_xnnpackfull_stridedgap_binsfirst.tflite` | 320 | 0.9844 | 1.21 / 2.67 px | 98.2% | 3.32 / 4.43 | 0.771 MB |
| `exported_tflite/doccornernet_v3_mnv3_224_int8_full_simcc_int8io_xnnpackfull_stridedgap_binsfirst.tflite` | 224 | 0.3519 | 50.43 / 129.92 px | 0.0% | 3.15 / 4.50 | 1.035 MB |

**Winner (overall): `mnv2_224_best` INT8**
- Best speed/accuracy tradeoff: fastest p50 (2.53ms) with `mean_iou=0.9888` (very close to the 256px model).
- If you want the top `mean_iou`, pick `mnv2_256_best` (slower due to 256px input).

### Float16 vs INT8 (per-model)

On MobileNetV2 models, **INT8 full-delegate is ~1.6–2.8× faster** with a very small `mean_iou` drop (≈0.0006–0.0011 absolute). On MobileNetV3, this PTQ INT8 export is **not usable** (accuracy collapses).

| Model | Float16 mean_iou | INT8 mean_iou | Δ mean_iou | Float16 p50 (ms) | INT8 p50 (ms) | Speedup |
|------|------------------:|--------------:|-----------:|-----------------:|--------------:|--------:|
| `mnv2_224_best` | 0.9894 | 0.9888 | -0.0006 | 4.24 | 2.53 | 1.67× |
| `mnv2_256_best` | 0.9902 | 0.9893 | -0.0009 | 8.18 | 2.92 | 2.80× |
| `mnv2_320` | 0.9855 | 0.9844 | -0.0011 | 5.36 | 3.32 | 1.61× |
| `mnv3_224` | 0.9842 | 0.3519 | -0.6324 | 3.96 | 3.15 | 1.26× |

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
├── export_tflite_int8.py           # PTQ export (int8/int16x8/dynamic)
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
