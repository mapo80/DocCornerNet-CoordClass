#!/bin/bash
# Sync training scripts to a Vast.ai instance (SSH/SCP).

set -euo pipefail

VAST_HOST="${VAST_HOST:-root@184.145.204.173}"
VAST_PORT="${VAST_PORT:-43091}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_docscanner_revlast}"
DEST_DIR="${DEST_DIR:-/root}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Files to transfer (training essentials)
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

SCP_OPTS=(
  -i "$SSH_KEY"
  -P "$VAST_PORT"
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
)

SSH_OPTS=(
  -i "$SSH_KEY"
  -p "$VAST_PORT"
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
)

echo "Transferring training scripts to Vast.ai..."
echo "Host: $VAST_HOST:$VAST_PORT"
echo "Dest: $DEST_DIR"
echo ""

ssh "${SSH_OPTS[@]}" "$VAST_HOST" "mkdir -p \"$DEST_DIR\""

for file in "${FILES[@]}"; do
  if [ -f "$SCRIPT_DIR/$file" ]; then
    echo "Uploading $file..."
    scp "${SCP_OPTS[@]}" "$SCRIPT_DIR/$file" "$VAST_HOST:$DEST_DIR/"
  else
    echo "Warning: $file not found, skipping"
  fi
done

echo ""
echo "Done! Files transferred to $VAST_HOST:$DEST_DIR"

