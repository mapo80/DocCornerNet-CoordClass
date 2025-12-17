"""
DocCornerNetV3: Heatmap-based SimCC for High-Precision Document Corner Detection.

Architecture:
- Backbone: MobileNetV3Small(alpha=0.75) - lightweight
- Neck: Mini-FPN combining features at 56x56
- Decoder: Bilinear upsampling to full 224x224 resolution
- Head: 4-channel heatmap at 224x224
- SimCC: Marginal logsumexp -> soft-argmax decode

Key difference from V2:
- NO cross-attention or corner queries
- Heatmap at FULL resolution (224x224) preserves spatial info
- Marginal distributions via logsumexp (not GAP)
- Expected value decode for sub-pixel precision

Target Performance:
- IoU >= 0.99 at 224x224
- Corner Error <= 1px mean
- Model size < 1M parameters
"""

from .model import (
    create_model,
    build_doccorner_simcc_v3,
    build_doccorner_simcc_v3_inference,
)

__version__ = "3.0.0"
