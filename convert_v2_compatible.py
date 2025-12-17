#!/usr/bin/env python3
"""
Convert DocCornerNetV3 to TFLite with V2-compatible format:
- Input: 'input' [1, 224, 224, 3]
- Output: 'Identity' [1, 9]
- No signatures
"""

import tensorflow as tf
import numpy as np

print(f"TensorFlow version: {tf.__version__}")

# Load the SavedModel
print("Loading SavedModel...")
loaded = tf.saved_model.load('savedmodel_inference')
infer_fn = loaded.signatures['serving_default']

# Create inference function with V2-compatible input name 'input'
@tf.function(input_signature=[tf.TensorSpec(shape=[1, 224, 224, 3], dtype=tf.float32, name='input')])
def inference(input):
    """Inference with V2-compatible naming"""
    result = infer_fn(image=input)
    coords = result['output_0']  # [B, 8]
    score_logit = result['output_1']  # [B, 1]
    score = tf.nn.sigmoid(score_logit)
    # Concat to [B, 9] like V2
    return tf.concat([coords, score], axis=-1)

# Get concrete function
print("Creating concrete function...")
concrete_func = inference.get_concrete_function()

# Test
print("\nTesting...")
test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
output = concrete_func(test_input)
print(f"Output shape: {output.shape}")
print(f"Output: {output.numpy()[0]}")

# Convert to TFLite - Float32
print("\n" + "="*60)
print("Converting to TFLite (float32)...")

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.experimental_new_converter = True
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open('model_float32.tflite', 'wb') as f:
    f.write(tflite_model)
print(f"Saved: model_float32.tflite ({len(tflite_model):,} bytes)")

# Float16
print("\nConverting to TFLite (float16)...")
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.experimental_new_converter = True
converter._experimental_lower_tensor_list_ops = False

tflite_model_fp16 = converter.convert()

with open('model_float16.tflite', 'wb') as f:
    f.write(tflite_model_fp16)
print(f"Saved: model_float16.tflite ({len(tflite_model_fp16):,} bytes)")

# Verify
print("\n" + "="*60)
print("Verifying models...")

for fname in ['model_float32.tflite', 'model_float16.tflite']:
    print(f"\n{fname}:")
    interpreter = tf.lite.Interpreter(model_path=fname)
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    print(f"  Input: {inp[0]['name']} {inp[0]['shape']}")
    print(f"  Output: {out[0]['name']} {out[0]['shape']}")
    print(f"  Signatures: {interpreter.get_signature_list()}")

# Compare with V2
print("\n" + "="*60)
print("Comparing with V2 model...")
interpreter_v2 = tf.lite.Interpreter(model_path='v2_model.tflite')
interpreter_v2.allocate_tensors()
v2_inp = interpreter_v2.get_input_details()
v2_out = interpreter_v2.get_output_details()
print(f"V2 Input: {v2_inp[0]['name']} {v2_inp[0]['shape']}")
print(f"V2 Output: {v2_out[0]['name']} {v2_out[0]['shape']}")
print(f"V2 Signatures: {interpreter_v2.get_signature_list()}")

print("\n" + "="*60)
print("DONE!")
