"""
Convert SavedModel to Keras .keras format for QAT compatibility.
Run this locally with the TF 2.18 environment where the model was trained.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np

print(f"TensorFlow: {tf.__version__}")

# Try to load the .keras model using model.py
import sys
sys.path.insert(0, '/Volumes/ZX20/ML-Models/DocScannerDetection/models/DocCornerNetV3')

from model import create_model

# Create model with same architecture
model = create_model(
    alpha=0.75,
    fpn_ch=48,
    simcc_ch=128,
    img_size=224,
    num_bins=224,
    tau=1.0,
)

# Load weights from .h5 file
model.load_weights('checkpoints/best_model.weights.h5')
print(f"Model: {model.count_params():,} params")

# Test inference
dummy = np.random.randn(1, 224, 224, 3).astype(np.float32)
out = model(dummy, training=False)
print(f"Outputs: {list(out.keys())}")

# Save as SavedModel with keras_metadata
print("\nSaving as SavedModel with Keras metadata...")
model.save('checkpoints/savedmodel_keras', save_format='tf')
print("Done!")

# Also save as H5
print("\nSaving entire model as H5...")
model.save('checkpoints/model_full.h5')
print("Done!")
