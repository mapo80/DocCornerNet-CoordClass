"""
Standalone QAT script that works in TensorFlow 2.15 Docker.

This script:
1. Defines the model architecture matching DocCornerNetV3
2. Loads weights from npz file
3. Applies QAT and fine-tunes
4. Exports to TFLite int8
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_npz", type=str, required=True,
                        help="Path to model_weights.npz")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=10,
                        help="QAT fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default="./qat_output")
    return parser.parse_args()


# ===================== Model Architecture =====================
# This exactly mirrors the DocCornerNetV3 model architecture

# Custom hard_swish activation (not available in all TF versions)
@tf.function
def hard_swish(x):
    return x * tf.nn.relu6(x + 3.0) / 6.0


# Register as custom activation
tf.keras.utils.get_custom_objects()['hard_swish'] = keras.layers.Activation(hard_swish)


class HardSwish(keras.layers.Layer):
    """Custom HardSwish activation layer."""
    def call(self, x):
        return x * tf.nn.relu6(x + 3.0) / 6.0


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _se_block(inputs, filters, se_ratio=0.25, prefix=''):
    x = keras.layers.GlobalAveragePooling2D(
        keepdims=True, name=f'{prefix}squeeze_excite_avg_pool')(inputs)
    x = keras.layers.Conv2D(
        _make_divisible(filters * se_ratio),
        kernel_size=1, padding='same',
        name=f'{prefix}squeeze_excite_conv')(x)
    x = keras.layers.Activation('relu', name=f'{prefix}squeeze_excite_relu')(x)
    x = keras.layers.Conv2D(
        filters, kernel_size=1, padding='same',
        name=f'{prefix}squeeze_excite_conv_1')(x)
    x = keras.layers.Activation('hard_sigmoid', name=f'{prefix}squeeze_excite_sigmoid')(x)
    return keras.layers.Multiply(name=f'{prefix}squeeze_excite_mul')([inputs, x])


def _get_activation_layer(activation, name):
    """Get activation layer handling hard_swish specially."""
    if activation == 'hard_swish':
        return HardSwish(name=name)
    return keras.layers.Activation(activation, name=name)


def _inverted_res_block(inputs, expansion, filters, kernel_size, stride,
                        se_ratio, activation, block_id):
    prefix = f'expanded_conv_{block_id}_' if block_id > 0 else 'expanded_conv_'
    infilters = inputs.shape[-1]

    x = inputs
    # Expand
    if block_id > 0:
        x = keras.layers.Conv2D(
            expansion, kernel_size=1, padding='same', use_bias=False,
            name=f'{prefix}expand')(x)
        x = keras.layers.BatchNormalization(
            epsilon=1e-3, momentum=0.999, name=f'{prefix}expand_bn')(x)
        x = _get_activation_layer(activation, f'{prefix}expand_act')(x)

    # Depthwise
    x = keras.layers.DepthwiseConv2D(
        kernel_size, strides=stride, padding='same', use_bias=False,
        name=f'{prefix}depthwise')(x)
    x = keras.layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name=f'{prefix}depthwise_bn')(x)
    x = _get_activation_layer(activation, f'{prefix}depthwise_act')(x)

    # SE block
    if se_ratio:
        x = _se_block(x, expansion if block_id > 0 else infilters, se_ratio, prefix)

    # Project
    x = keras.layers.Conv2D(
        filters, kernel_size=1, padding='same', use_bias=False,
        name=f'{prefix}project')(x)
    x = keras.layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name=f'{prefix}project_bn')(x)

    # Residual
    if stride == 1 and infilters == filters:
        x = keras.layers.Add(name=f'{prefix}add')([inputs, x])

    return x


def create_mobilenetv3_small(inputs, alpha=0.75):
    """MobileNetV3-Small backbone."""

    def depth(d):
        return _make_divisible(d * alpha)

    # Initial conv
    x = keras.layers.Conv2D(
        16, kernel_size=3, strides=2, padding='same', use_bias=False,
        name='conv')(inputs)
    x = keras.layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='conv_bn')(x)
    x = HardSwish(name='conv_act')(x)

    # MobileNetV3-Small blocks
    # Block 0
    x = _inverted_res_block(x, 16, depth(16), 3, 2, 0.25, 'relu', 0)
    c2 = x  # stride 4

    # Block 1-2
    x = _inverted_res_block(x, 72, depth(24), 3, 2, None, 'relu', 1)
    x = _inverted_res_block(x, 88, depth(24), 3, 1, None, 'relu', 2)
    c3 = x  # stride 8

    # Block 3-7
    x = _inverted_res_block(x, 96, depth(40), 5, 2, 0.25, 'hard_swish', 3)
    x = _inverted_res_block(x, 240, depth(40), 5, 1, 0.25, 'hard_swish', 4)
    x = _inverted_res_block(x, 240, depth(40), 5, 1, 0.25, 'hard_swish', 5)
    x = _inverted_res_block(x, 120, depth(48), 5, 1, 0.25, 'hard_swish', 6)
    x = _inverted_res_block(x, 144, depth(48), 5, 1, 0.25, 'hard_swish', 7)
    c4 = x  # stride 16

    # Block 8-10
    x = _inverted_res_block(x, 288, depth(96), 5, 2, 0.25, 'hard_swish', 8)
    x = _inverted_res_block(x, 576, depth(96), 5, 1, 0.25, 'hard_swish', 9)
    x = _inverted_res_block(x, 576, depth(96), 5, 1, 0.25, 'hard_swish', 10)

    # Final conv
    x = keras.layers.Conv2D(
        576, kernel_size=1, padding='same', use_bias=False,
        name='conv_1')(x)
    x = keras.layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='conv_1_bn')(x)
    x = HardSwish(name='conv_1_act')(x)
    c5 = x  # stride 32

    return {'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5}


def create_fpn(features, fpn_ch=48):
    """FPN neck."""
    c2, c3, c4, c5 = features['c2'], features['c3'], features['c4'], features['c5']

    # Lateral connections with BN
    p4 = keras.layers.Conv2D(fpn_ch, 1, padding='same', name='fpn_lat_c4')(c4)
    p4 = keras.layers.BatchNormalization(name='fpn_lat_c4_bn')(p4)

    p3 = keras.layers.Conv2D(fpn_ch, 1, padding='same', name='fpn_lat_c3')(c3)
    p3 = keras.layers.BatchNormalization(name='fpn_lat_c3_bn')(p3)

    p2 = keras.layers.Conv2D(fpn_ch, 1, padding='same', name='fpn_lat_c2')(c2)
    p2 = keras.layers.BatchNormalization(name='fpn_lat_c2_bn')(p2)

    # Top-down pathway with refinement
    p4_up = keras.layers.UpSampling2D(2, name='fpn_up_p4')(p4)
    p3 = keras.layers.Add(name='fpn_add_p3')([p3, p4_up])
    p3 = keras.layers.SeparableConv2D(fpn_ch, 3, padding='same', name='fpn_p3_refine_sepconv')(p3)
    p3 = keras.layers.BatchNormalization(name='fpn_p3_refine_bn')(p3)
    p3 = keras.layers.Activation('relu', name='fpn_p3_refine_act')(p3)

    p3_up = keras.layers.UpSampling2D(2, name='fpn_up_p3')(p3)
    p2 = keras.layers.Add(name='fpn_add_p2')([p2, p3_up])
    p2 = keras.layers.SeparableConv2D(fpn_ch, 3, padding='same', name='fpn_p2_refine_sepconv')(p2)
    p2 = keras.layers.BatchNormalization(name='fpn_p2_refine_bn')(p2)
    p2 = keras.layers.Activation('relu', name='fpn_p2_refine_act')(p2)

    # Combine P2 and P3 for richer features
    p3_to_p2 = keras.layers.UpSampling2D(2, name='fpn_p3_to_p2')(p3)
    combined = keras.layers.Concatenate(name='fpn_combine')([p2, p3_to_p2])

    # Refine combined features
    combined = keras.layers.SeparableConv2D(96, 3, padding='same', name='simcc_refine1_sepconv')(combined)
    combined = keras.layers.BatchNormalization(name='simcc_refine1_bn')(combined)
    combined = keras.layers.Activation('relu', name='simcc_refine1_act')(combined)

    combined = keras.layers.SeparableConv2D(48, 3, padding='same', name='simcc_refine2_sepconv')(combined)
    combined = keras.layers.BatchNormalization(name='simcc_refine2_bn')(combined)
    combined = keras.layers.Activation('relu', name='simcc_refine2_act')(combined)

    return combined


def create_simcc_head(features, num_bins=224, simcc_ch=128):
    """SimCC head with marginal pooling."""

    # Marginal pooling - X direction (pool along height)
    x_pool = keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1),
        name='marginal_x_pool')(features)  # [B, W, C]

    # Marginal pooling - Y direction (pool along width)
    y_pool = keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=2),
        name='marginal_y_pool')(features)  # [B, H, C]

    # Global context
    global_feat = keras.layers.GlobalAveragePooling2D(name='simcc_global_pool')(features)
    global_feat = keras.layers.Dense(64, name='simcc_global_fc')(global_feat)

    # SimCC X branch
    x_feat = keras.layers.Conv1D(simcc_ch, 5, padding='same', name='simcc_x_conv1')(x_pool)
    x_feat = keras.layers.BatchNormalization(name='simcc_x_bn1')(x_feat)
    x_feat = keras.layers.Activation('relu', name='simcc_x_act1')(x_feat)
    x_feat = keras.layers.Conv1D(64, 3, padding='same', name='simcc_x_conv2')(x_feat)
    x_feat = keras.layers.BatchNormalization(name='simcc_x_bn2')(x_feat)
    x_feat = keras.layers.Activation('relu', name='simcc_x_act2')(x_feat)

    # Add global context to X
    global_x = keras.layers.RepeatVector(x_feat.shape[1])(global_feat)
    x_feat = keras.layers.Concatenate(name='simcc_x_cat')([x_feat, global_x])
    simcc_x = keras.layers.Conv1D(4, 1, name='simcc_x_out')(x_feat)  # [B, W, 4]
    simcc_x = keras.layers.Permute((2, 1), name='simcc_x')(simcc_x)  # [B, 4, W]

    # SimCC Y branch
    y_feat = keras.layers.Conv1D(simcc_ch, 5, padding='same', name='simcc_y_conv1')(y_pool)
    y_feat = keras.layers.BatchNormalization(name='simcc_y_bn1')(y_feat)
    y_feat = keras.layers.Activation('relu', name='simcc_y_act1')(y_feat)
    y_feat = keras.layers.Conv1D(64, 3, padding='same', name='simcc_y_conv2')(y_feat)
    y_feat = keras.layers.BatchNormalization(name='simcc_y_bn2')(y_feat)
    y_feat = keras.layers.Activation('relu', name='simcc_y_act2')(y_feat)

    # Add global context to Y
    global_y = keras.layers.RepeatVector(y_feat.shape[1])(global_feat)
    y_feat = keras.layers.Concatenate(name='simcc_y_cat')([y_feat, global_y])
    simcc_y = keras.layers.Conv1D(4, 1, name='simcc_y_out')(y_feat)  # [B, H, 4]
    simcc_y = keras.layers.Permute((2, 1), name='simcc_y')(simcc_y)  # [B, 4, H]

    return simcc_x, simcc_y


def soft_argmax(logits, tau=1.0):
    """Decode coordinates using soft-argmax."""
    num_bins = tf.cast(tf.shape(logits)[-1], tf.float32)
    bins = tf.range(num_bins, dtype=tf.float32) / num_bins
    bins = tf.reshape(bins, [1, 1, -1])

    probs = tf.nn.softmax(logits / tau, axis=-1)
    coords = tf.reduce_sum(probs * bins, axis=-1)
    return coords


def create_model(alpha=0.75, fpn_ch=48, simcc_ch=128, num_bins=224, tau=1.0):
    """Create full DocCornerNetV3 model."""

    inputs = keras.Input(shape=(224, 224, 3), name='image')

    # Normalize to [-1, 1]
    x = keras.layers.Rescaling(1./127.5, offset=-1, name='rescaling')(inputs)

    # Backbone
    features = create_mobilenetv3_small(x, alpha=alpha)

    # FPN
    fpn_out = create_fpn(features, fpn_ch=fpn_ch)

    # SimCC heads
    simcc_x, simcc_y = create_simcc_head(fpn_out, num_bins=num_bins, simcc_ch=simcc_ch)

    # Decode coordinates
    coords_x = keras.layers.Lambda(
        lambda t: soft_argmax(t, tau), name='decode_x')(simcc_x)  # [B, 4]
    coords_y = keras.layers.Lambda(
        lambda t: soft_argmax(t, tau), name='decode_y')(simcc_y)  # [B, 4]

    # Interleave x and y
    coords = keras.layers.Lambda(
        lambda xy: tf.stack([
            xy[0][:, 0], xy[1][:, 0],
            xy[0][:, 1], xy[1][:, 1],
            xy[0][:, 2], xy[1][:, 2],
            xy[0][:, 3], xy[1][:, 3]
        ], axis=-1),
        name='coords'
    )([coords_x, coords_y])

    # Score head
    c5 = features['c5']
    score_gap = keras.layers.GlobalAveragePooling2D(name='score_gap')(c5)
    score_logit = keras.layers.Dense(1, name='score_logit')(score_gap)

    model = keras.Model(
        inputs=inputs,
        outputs={
            'simcc_x': simcc_x,
            'simcc_y': simcc_y,
            'coords': coords,
            'score_logit': score_logit
        },
        name='DocCornerNetV3'
    )

    return model


def load_weights_from_npz(model, npz_path):
    """Load weights from npz file."""
    data = np.load(npz_path, allow_pickle=True)

    loaded = 0
    failed = 0

    for layer in model.layers:
        if layer.name in data:
            weights = data[layer.name]
            # Convert from object array to list of numpy arrays
            weight_list = [np.array(w) for w in weights]
            try:
                layer.set_weights(weight_list)
                loaded += 1
            except Exception as e:
                print(f"Failed to load weights for {layer.name}: {e}")
                failed += 1

    print(f"Loaded weights for {loaded} layers, failed for {failed} layers")


# ===================== Data Loading =====================

def create_dataset(data_root, split, batch_size, img_size=224, shuffle=True):
    """Create tf.data.Dataset."""
    data_root = Path(data_root)
    split_file = data_root / f"{split}.txt"

    with open(split_file) as f:
        filenames = [line.strip() for line in f if line.strip()]

    images_dir = data_root / "images"
    labels_dir = data_root / "labels"

    # Build data
    image_paths = []
    coords_list = []

    for fname in filenames:
        img_path = images_dir / fname
        lbl_path = labels_dir / fname.replace('.jpg', '.txt').replace('.png', '.txt')

        if not img_path.exists() or not lbl_path.exists():
            continue

        with open(lbl_path) as f:
            line = f.readline().strip()

        if not line:
            continue

        parts = line.split()
        if len(parts) < 9:
            continue

        coords = [float(x) for x in parts[1:9]]
        image_paths.append(str(img_path))
        coords_list.append(coords)

    print(f"Loaded {len(image_paths)} samples for {split}")

    def generator():
        indices = list(range(len(image_paths)))
        if shuffle:
            np.random.shuffle(indices)

        for i in indices:
            img = tf.io.read_file(image_paths[i])
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [img_size, img_size])
            img = tf.cast(img, tf.float32) / 255.0

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std

            coords = np.array(coords_list[i], dtype=np.float32)
            has_doc = np.array([1.0], dtype=np.float32)

            yield img, {'coords': coords, 'has_document': has_doc}

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
            {
                'coords': tf.TensorSpec(shape=(8,), dtype=tf.float32),
                'has_document': tf.TensorSpec(shape=(1,), dtype=tf.float32)
            }
        )
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ===================== Loss =====================

def create_loss_fn(sigma_px=3.0, num_bins=224):
    """Create combined loss function."""

    def simcc_loss(y_true, y_pred):
        # y_true: [B, 4] - normalized coords
        # y_pred: [B, 4, num_bins] - logits
        bins = tf.range(num_bins, dtype=tf.float32) / num_bins
        bins = tf.reshape(bins, [1, 1, -1])

        y_true_exp = tf.expand_dims(y_true, -1)
        sigma_norm = sigma_px / num_bins
        targets = tf.exp(-0.5 * tf.square((bins - y_true_exp) / sigma_norm))
        targets = targets / (tf.reduce_sum(targets, axis=-1, keepdims=True) + 1e-8)

        pred_prob = tf.nn.softmax(y_pred, axis=-1)
        loss = -tf.reduce_sum(targets * tf.math.log(pred_prob + 1e-8), axis=-1)
        return tf.reduce_mean(loss)

    def loss_fn(y_true, y_pred):
        gt_coords = y_true['coords']
        gt_has_doc = y_true['has_document']

        simcc_x = y_pred['simcc_x']
        simcc_y = y_pred['simcc_y']
        pred_coords = y_pred['coords']
        score_logit = y_pred['score_logit']

        # Extract X and Y
        gt_x = tf.stack([gt_coords[:, 0], gt_coords[:, 2], gt_coords[:, 4], gt_coords[:, 6]], axis=1)
        gt_y = tf.stack([gt_coords[:, 1], gt_coords[:, 3], gt_coords[:, 5], gt_coords[:, 7]], axis=1)

        # SimCC losses
        loss_x = simcc_loss(gt_x, simcc_x)
        loss_y = simcc_loss(gt_y, simcc_y)
        loss_simcc = (loss_x + loss_y) / 2.0

        # Coord L1 loss
        mask = tf.squeeze(gt_has_doc, -1)
        coord_diff = tf.abs(pred_coords - gt_coords)
        loss_coord = tf.reduce_sum(coord_diff * tf.expand_dims(mask, -1)) / (tf.reduce_sum(mask) * 8.0 + 1e-8)

        # Score BCE
        loss_score = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_has_doc, logits=score_logit)
        )

        return loss_simcc + 0.5 * loss_coord + 0.5 * loss_score

    return loss_fn


# ===================== Main =====================

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("QAT Training (Standalone)")
    print("=" * 60)
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU: {tf.config.list_physical_devices('GPU')}")

    # Create model
    print("\nCreating model...")
    model = create_model(alpha=0.75, fpn_ch=48, simcc_ch=128, num_bins=224, tau=1.0)
    print(f"Model params: {model.count_params():,}")

    # Load weights
    print(f"\nLoading weights from {args.weights_npz}...")
    load_weights_from_npz(model, args.weights_npz)

    # Test inference
    dummy = np.random.randn(1, 224, 224, 3).astype(np.float32)
    out = model(dummy, training=False)
    print(f"Test inference: coords shape = {out['coords'].shape}")

    # Apply QAT
    print("\nApplying QAT...")
    import tensorflow_model_optimization as tfmot
    print(f"tfmot version: {tfmot.__version__}")

    qat_model = tfmot.quantization.keras.quantize_model(model)
    print(f"QAT model params: {qat_model.count_params():,}")

    # Create datasets
    print("\nLoading datasets...")
    train_ds = create_dataset(args.data_root, 'train', args.batch_size)
    val_ds = create_dataset(args.data_root, 'val', args.batch_size, shuffle=False)

    # Compile
    loss_fn = create_loss_fn()
    qat_model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr), loss=loss_fn)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Training
        train_losses = []
        for batch_idx, (images, targets) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                preds = qat_model(images, training=True)
                loss = loss_fn(targets, preds)

            grads = tape.gradient(loss, qat_model.trainable_variables)
            qat_model.optimizer.apply_gradients(zip(grads, qat_model.trainable_variables))
            train_losses.append(float(loss))

            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}: loss = {np.mean(train_losses[-50:]):.4f}")

        # Validation
        val_losses = []
        for images, targets in val_ds:
            preds = qat_model(images, training=False)
            loss = loss_fn(targets, preds)
            val_losses.append(float(loss))

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        print(f"  train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            qat_model.save(str(output_dir / "qat_best.keras"))
            print(f"  Saved best model")

    # Save final
    qat_model.save(str(output_dir / "qat_final.keras"))

    # Export to TFLite int8
    print("\nExporting to TFLite int8...")

    def representative_dataset():
        for images, _ in val_ds.take(100):
            for i in range(images.shape[0]):
                yield [images[i:i+1].numpy()]

    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    try:
        tflite_model = converter.convert()
        tflite_path = output_dir / "model_qat_int8.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Saved: {tflite_path} ({len(tflite_model)/(1024*1024):.2f} MB)")
    except Exception as e:
        print(f"TFLite conversion failed: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
