#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${DATA_ROOT:-/Volumes/ZX20/ML-Models/DocScannerDetection/datasets/official/doc-scanner-dataset-rev-new}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train_clean_iter3_plus_hard_full}"
VAL_SPLIT="${VAL_SPLIT:-val_clean_iter3}"

INIT_WEIGHTS="${INIT_WEIGHTS:-checkpoints/mobilenetv2_256_clean_iter3}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mobilenetv2_224_from256_clean_iter3}"

BACKBONE="${BACKBONE:-mobilenetv2}"
ALPHA="${ALPHA:-0.35}"
IMG_SIZE="${IMG_SIZE:-224}"
NUM_BINS="${NUM_BINS:-224}"
FPN_CH="${FPN_CH:-32}"
SIMCC_CH="${SIMCC_CH:-96}"

# Mac default: keep this conservative; override via env BATCH_SIZE=...
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-60}"
LR="${LR:-3e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
PATIENCE="${PATIENCE:-8}"
LR_PATIENCE="${LR_PATIENCE:-3}"
LR_FACTOR="${LR_FACTOR:-0.5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-16}"

LOG="checkpoints/training_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p checkpoints

echo "DATA_ROOT=$DATA_ROOT"
echo "TRAIN_SPLIT=$TRAIN_SPLIT"
echo "VAL_SPLIT=$VAL_SPLIT"
echo "INIT_WEIGHTS=$INIT_WEIGHTS"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "LOG=$LOG"
echo

python3 -u train_ultra.py \
  --data_root "$DATA_ROOT" \
  --train_split "$TRAIN_SPLIT" \
  --val_split "$VAL_SPLIT" \
  --output_dir checkpoints \
  --experiment_name "$EXPERIMENT_NAME" \
  --backbone "$BACKBONE" \
  --alpha "$ALPHA" \
  --img_size "$IMG_SIZE" \
  --num_bins "$NUM_BINS" \
  --fpn_ch "$FPN_CH" \
  --simcc_ch "$SIMCC_CH" \
  --backbone_weights none \
  --init_weights "$INIT_WEIGHTS" \
  --init_partial \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --patience "$PATIENCE" \
  --lr_patience "$LR_PATIENCE" \
  --lr_factor "$LR_FACTOR" \
  --num_workers "$NUM_WORKERS" \
  --augment \
  2>&1 | tee "$LOG"
