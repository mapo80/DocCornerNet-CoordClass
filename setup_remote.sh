#!/bin/bash
# =============================================================================
# DocCornerNet - Remote Machine Setup Script
# =============================================================================
# This script sets up a remote machine (e.g., RunPod, Lambda Labs, etc.) for
# training DocCornerNet models. It handles:
# - Git clone/pull of the repository
# - Python dependencies installation
# - Optional HuggingFace dataset download
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/mapo80/DocCornerNet-CoordClass/main/setup_remote.sh | bash
#   # or
#   wget -qO- https://raw.githubusercontent.com/mapo80/DocCornerNet-CoordClass/main/setup_remote.sh | bash
#   # or (after cloning)
#   bash setup_remote.sh [--download-dataset] [--hf-token TOKEN]
#
# Options:
#   --download-dataset    Download the HuggingFace dataset after setup
#   --hf-token TOKEN      HuggingFace token for private datasets (optional)
#   --output-dir DIR      Directory for dataset download (default: ./hf_dataset)
#   --branch BRANCH       Git branch to checkout (default: main)
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
REPO_URL="https://github.com/mapo80/DocCornerNet-CoordClass.git"
REPO_DIR="/root/DocCornerNet-CoordClass"
BRANCH="main"
DOWNLOAD_DATASET=false
HF_TOKEN=""
OUTPUT_DIR="./hf_dataset"
HF_DATASET="mapo80/DocCornerDataset"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --download-dataset)
            DOWNLOAD_DATASET=true
            shift
            ;;
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --branch)
            BRANCH="$2"
            shift 2
            ;;
        --repo-dir)
            REPO_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --download-dataset    Download the HuggingFace dataset after setup"
            echo "  --hf-token TOKEN      HuggingFace token for private datasets"
            echo "  --output-dir DIR      Directory for dataset download (default: ./hf_dataset)"
            echo "  --branch BRANCH       Git branch to checkout (default: main)"
            echo "  --repo-dir DIR        Directory for repository (default: /root/DocCornerNet-CoordClass)"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DocCornerNet Remote Setup${NC}"
echo -e "${BLUE}========================================${NC}"

# -----------------------------------------------------------------------------
# Step 1: Clone or update repository
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[1/4] Setting up repository...${NC}"

if [ -d "$REPO_DIR/.git" ]; then
    echo -e "Repository exists, pulling latest changes..."
    cd "$REPO_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo -e "Cloning repository..."
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

echo -e "${GREEN}Repository ready at: $REPO_DIR${NC}"

# -----------------------------------------------------------------------------
# Step 2: Install Python dependencies
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/4] Installing Python dependencies...${NC}"

# Check Python version
python3 --version || { echo -e "${RED}Python3 not found!${NC}"; exit 1; }

# Upgrade pip
pip install --upgrade pip

# Core dependencies
echo -e "Installing core dependencies..."
pip install \
    tensorflow[and-cuda] \
    numpy \
    pillow \
    tqdm \
    opencv-python-headless \
    shapely

# HuggingFace dependencies (for dataset download)
echo -e "Installing HuggingFace dependencies..."
pip install \
    datasets \
    huggingface_hub \
    pyarrow

echo -e "${GREEN}Dependencies installed successfully${NC}"

# -----------------------------------------------------------------------------
# Step 3: Verify installation
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[3/4] Verifying installation...${NC}"

python3 -c "
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

print(f'TensorFlow: {tf.__version__}')
print(f'NumPy: {np.__version__}')
print(f'OpenCV: {cv2.__version__}')

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'GPUs available: {len(gpus)}')
    for gpu in gpus:
        print(f'  - {gpu.name}')
else:
    print('No GPU detected (will use CPU)')
"

echo -e "${GREEN}Installation verified${NC}"

# -----------------------------------------------------------------------------
# Step 4: Download dataset (optional)
# -----------------------------------------------------------------------------
if [ "$DOWNLOAD_DATASET" = true ]; then
    echo -e "\n${YELLOW}[4/4] Downloading HuggingFace dataset...${NC}"

    cd "$REPO_DIR"

    # Build download command
    DOWNLOAD_CMD="python train_ultra.py --hf_dataset $HF_DATASET --download_hf $OUTPUT_DIR"

    if [ -n "$HF_TOKEN" ]; then
        DOWNLOAD_CMD="$DOWNLOAD_CMD --hf_token $HF_TOKEN"
    fi

    echo -e "Running: $DOWNLOAD_CMD"
    eval "$DOWNLOAD_CMD"

    echo -e "${GREEN}Dataset downloaded to: $OUTPUT_DIR${NC}"
else
    echo -e "\n${YELLOW}[4/4] Skipping dataset download${NC}"
    echo -e "To download later, run:"
    echo -e "  python train_ultra.py --hf_dataset $HF_DATASET --download_hf ./hf_dataset"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e ""
echo -e "Repository: $REPO_DIR"
echo -e ""
echo -e "${YELLOW}Quick start commands:${NC}"
echo -e ""
echo -e "# Download dataset (if not done already)"
echo -e "python train_ultra.py --hf_dataset mapo80/DocCornerDataset --download_hf ./hf_dataset"
echo -e ""
echo -e "# Train mobile model (alpha=0.35, 256px)"
echo -e "python train_ultra.py \\"
echo -e "    --hf_dataset ./hf_dataset \\"
echo -e "    --output_dir ./checkpoints \\"
echo -e "    --backbone mobilenetv2 \\"
echo -e "    --alpha 0.35 \\"
echo -e "    --img_size 256 \\"
echo -e "    --num_bins 256 \\"
echo -e "    --batch_size 512 \\"
echo -e "    --epochs 200"
echo -e ""
echo -e "# Train server model (alpha=1.0, 320px)"
echo -e "python train_ultra.py \\"
echo -e "    --hf_dataset ./hf_dataset \\"
echo -e "    --output_dir ./checkpoints \\"
echo -e "    --backbone mobilenetv2 \\"
echo -e "    --alpha 1.0 \\"
echo -e "    --img_size 320 \\"
echo -e "    --num_bins 320 \\"
echo -e "    --simcc_ch 128 \\"
echo -e "    --fpn_ch 48 \\"
echo -e "    --batch_size 128 \\"
echo -e "    --epochs 200"
echo -e ""
