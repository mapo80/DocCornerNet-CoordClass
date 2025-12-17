"""
Test QAT loading in Docker environment.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

# Load model from SavedModel
print("\nLoading model from SavedModel...")
model = tf.keras.models.load_model('/workspace/checkpoints/savedmodel_inference', compile=False)
print(f"Model loaded: {model.count_params():,} params")

# Test inference
import numpy as np
dummy = np.random.randn(1, 224, 224, 3).astype(np.float32)
out = model(dummy, training=False)
print(f"Outputs: {list(out.keys())}")
for k, v in out.items():
    print(f"  {k}: {v.shape}")

# Apply QAT
print("\nApplying QAT...")
import tensorflow_model_optimization as tfmot
print(f"tfmot: {tfmot.__version__}")

qat_model = tfmot.quantization.keras.quantize_model(model)
print(f"QAT model: {qat_model.count_params():,} params")

# Test QAT inference
out_qat = qat_model(dummy, training=False)
print(f"QAT outputs: {list(out_qat.keys())}")

print("\nSUCCESS!")
