#!/bin/bash
# Sync training scripts to RunPod server

RUNPOD_HOST="${RUNPOD_HOST:-root@213.173.105.7}"
RUNPOD_PORT="${RUNPOD_PORT:-30272}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
DEST_DIR="${DEST_DIR:-/root}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Files to transfer
FILES=(
    "train.py"
    "train_optimized.py"
    "train_ultra.py"
    "dataset.py"
    "model.py"
    "losses.py"
    "metrics.py"
    "evaluate.py"
    "generate_full_eval_csv.py"
)

echo "Transferring training scripts to RunPod..."
echo "Host: $RUNPOD_HOST:$RUNPOD_PORT"
echo ""

for file in "${FILES[@]}"; do
    if [ -f "$SCRIPT_DIR/$file" ]; then
        echo "Uploading $file..."
        scp -i $SSH_KEY -P $RUNPOD_PORT -o StrictHostKeyChecking=no "$SCRIPT_DIR/$file" "$RUNPOD_HOST:$DEST_DIR/"
    else
        echo "Warning: $file not found, skipping"
    fi
done

echo ""
echo "Done! Files transferred to $RUNPOD_HOST:$DEST_DIR"
