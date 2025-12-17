"""
QAT Training script for Docker (TensorFlow 2.15 with Keras 2.x).

This recreates the model architecture and loads weights from .h5 file
to ensure compatibility with tensorflow-model-optimization.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--saved_model", type=str, required=True,
                        help="Path to SavedModel directory")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_qat")
    return parser.parse_args()


# ============ Model Architecture (must match original) ============

def hard_swish(x):
    return x * tf.nn.relu6(x + 3.0) / 6.0


def create_se_block(x, filters, ratio=4, name=""):
    """Squeeze-and-Excitation block."""
    se = keras.layers.GlobalAveragePooling2D(name=f"{name}_se_gap")(x)
    se = keras.layers.Reshape((1, 1, filters), name=f"{name}_se_reshape")(se)
    se = keras.layers.Conv2D(filters // ratio, 1, padding='same', name=f"{name}_se_reduce")(se)
    se = keras.layers.Activation('relu', name=f"{name}_se_relu")(se)
    se = keras.layers.Conv2D(filters, 1, padding='same', name=f"{name}_se_expand")(se)
    se = keras.layers.Activation('hard_sigmoid', name=f"{name}_se_sigmoid")(se)
    return keras.layers.Multiply(name=f"{name}_se_mul")([x, se])


def inverted_residual_block(x, expand_ch, out_ch, kernel, stride, use_se=False, activation='relu', name=""):
    """MobileNetV3 inverted residual block."""
    in_ch = x.shape[-1]

    # Expand
    if expand_ch != in_ch:
        expanded = keras.layers.Conv2D(expand_ch, 1, padding='same', use_bias=False, name=f"{name}_expand")(x)
        expanded = keras.layers.BatchNormalization(name=f"{name}_expand_bn")(expanded)
        expanded = keras.layers.Activation(activation, name=f"{name}_expand_act")(expanded)
    else:
        expanded = x

    # Depthwise
    dw = keras.layers.DepthwiseConv2D(kernel, stride, padding='same', use_bias=False, name=f"{name}_dw")(expanded)
    dw = keras.layers.BatchNormalization(name=f"{name}_dw_bn")(dw)
    dw = keras.layers.Activation(activation, name=f"{name}_dw_act")(dw)

    # SE block
    if use_se:
        dw = create_se_block(dw, expand_ch, name=name)

    # Project
    out = keras.layers.Conv2D(out_ch, 1, padding='same', use_bias=False, name=f"{name}_project")(dw)
    out = keras.layers.BatchNormalization(name=f"{name}_project_bn")(out)

    # Residual
    if stride == 1 and in_ch == out_ch:
        out = keras.layers.Add(name=f"{name}_add")([x, out])

    return out


def create_mobilenetv3_small_backbone(input_tensor, alpha=0.75):
    """Create MobileNetV3-Small backbone (simplified version for QAT)."""

    def _depth(d, divisor=8):
        return max(divisor, int(d * alpha + divisor / 2) // divisor * divisor)

    # Initial conv
    x = keras.layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False, name='conv')(input_tensor)
    x = keras.layers.BatchNormalization(name='conv_bn')(x)
    x = keras.layers.Activation('hard_swish', name='conv_act')(x)

    # MobileNetV3-Small blocks (simplified)
    # Block 1: stride 2
    x = inverted_residual_block(x, 16, _depth(16), 3, 2, use_se=True, activation='relu', name='block1')
    c2 = x  # 56x56

    # Block 2-3: stride 2
    x = inverted_residual_block(x, 72, _depth(24), 3, 2, use_se=False, activation='relu', name='block2')
    x = inverted_residual_block(x, 88, _depth(24), 3, 1, use_se=False, activation='relu', name='block3')
    c3 = x  # 28x28

    # Block 4-7: stride 2
    x = inverted_residual_block(x, 96, _depth(40), 5, 2, use_se=True, activation='hard_swish', name='block4')
    x = inverted_residual_block(x, 240, _depth(40), 5, 1, use_se=True, activation='hard_swish', name='block5')
    x = inverted_residual_block(x, 240, _depth(40), 5, 1, use_se=True, activation='hard_swish', name='block6')
    x = inverted_residual_block(x, 120, _depth(48), 5, 1, use_se=True, activation='hard_swish', name='block7')
    c4 = x  # 14x14

    # Block 8-11: stride 2
    x = inverted_residual_block(x, 144, _depth(48), 5, 1, use_se=True, activation='hard_swish', name='block8')
    x = inverted_residual_block(x, 288, _depth(96), 5, 2, use_se=True, activation='hard_swish', name='block9')
    x = inverted_residual_block(x, 576, _depth(96), 5, 1, use_se=True, activation='hard_swish', name='block10')
    x = inverted_residual_block(x, 576, _depth(96), 5, 1, use_se=True, activation='hard_swish', name='block11')
    c5 = x  # 7x7

    return {'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5}


def create_fpn(features, fpn_ch=48):
    """Create FPN for multi-scale features."""
    c2, c3, c4, c5 = features['c2'], features['c3'], features['c4'], features['c5']

    # Lateral connections
    p5 = keras.layers.Conv2D(fpn_ch, 1, padding='same', name='fpn_p5')(c5)
    p4 = keras.layers.Conv2D(fpn_ch, 1, padding='same', name='fpn_p4')(c4)
    p3 = keras.layers.Conv2D(fpn_ch, 1, padding='same', name='fpn_p3')(c3)
    p2 = keras.layers.Conv2D(fpn_ch, 1, padding='same', name='fpn_p2')(c2)

    # Top-down pathway
    p4 = keras.layers.Add(name='fpn_add_p4')([
        p4, keras.layers.UpSampling2D(2, name='fpn_up_p5')(p5)
    ])
    p3 = keras.layers.Add(name='fpn_add_p3')([
        p3, keras.layers.UpSampling2D(2, name='fpn_up_p4')(p4)
    ])
    p2 = keras.layers.Add(name='fpn_add_p2')([
        p2, keras.layers.UpSampling2D(2, name='fpn_up_p3')(p3)
    ])

    return p2  # 56x56 x fpn_ch


def create_simcc_head(x, num_bins=224, simcc_ch=128):
    """Create SimCC coordinate prediction head."""

    # Marginal pooling for X
    x_pool = keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1, keepdims=True),
        name='marginal_pool_x'
    )(x)  # [B, 1, W, C]
    x_pool = keras.layers.Reshape((-1, x.shape[-1]), name='reshape_x')(x_pool)

    # Marginal pooling for Y
    y_pool = keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=2, keepdims=True),
        name='marginal_pool_y'
    )(x)  # [B, H, 1, C]
    y_pool = keras.layers.Reshape((-1, x.shape[-1]), name='reshape_y')(y_pool)

    # SimCC X branch
    simcc_x = keras.layers.Dense(simcc_ch, activation='relu', name='simcc_x_fc1')(x_pool)
    simcc_x = keras.layers.Dense(num_bins * 4, name='simcc_x_fc2')(simcc_x)
    simcc_x = keras.layers.Reshape((num_bins, 4), name='simcc_x_reshape')(simcc_x)
    simcc_x = keras.layers.Permute((2, 1), name='simcc_x')(simcc_x)  # [B, 4, num_bins]

    # SimCC Y branch
    simcc_y = keras.layers.Dense(simcc_ch, activation='relu', name='simcc_y_fc1')(y_pool)
    simcc_y = keras.layers.Dense(num_bins * 4, name='simcc_y_fc2')(simcc_y)
    simcc_y = keras.layers.Reshape((num_bins, 4), name='simcc_y_reshape')(simcc_y)
    simcc_y = keras.layers.Permute((2, 1), name='simcc_y')(simcc_y)  # [B, 4, num_bins]

    return simcc_x, simcc_y


def decode_coords(simcc_x, simcc_y, tau=1.0):
    """Decode coordinates from SimCC logits using soft-argmax."""
    num_bins = tf.cast(tf.shape(simcc_x)[-1], tf.float32)

    # Apply temperature and softmax
    prob_x = tf.nn.softmax(simcc_x / tau, axis=-1)
    prob_y = tf.nn.softmax(simcc_y / tau, axis=-1)

    # Create bin indices
    bins = tf.range(num_bins, dtype=tf.float32) / num_bins
    bins = tf.reshape(bins, [1, 1, -1])

    # Soft-argmax
    coord_x = tf.reduce_sum(prob_x * bins, axis=-1)  # [B, 4]
    coord_y = tf.reduce_sum(prob_y * bins, axis=-1)  # [B, 4]

    # Interleave x and y: [x0, y0, x1, y1, x2, y2, x3, y3]
    coords = tf.stack([
        coord_x[:, 0], coord_y[:, 0],
        coord_x[:, 1], coord_y[:, 1],
        coord_x[:, 2], coord_y[:, 2],
        coord_x[:, 3], coord_y[:, 3]
    ], axis=-1)

    return coords


def create_model(alpha=0.75, fpn_ch=48, simcc_ch=128, num_bins=224, img_size=224, tau=1.0):
    """Create full DocCornerNetV3 model."""

    # Input
    inputs = keras.Input(shape=(img_size, img_size, 3), name='image')

    # Normalize to [-1, 1]
    x = keras.layers.Rescaling(1./127.5, offset=-1, name='rescaling')(inputs)

    # Backbone
    features = create_mobilenetv3_small_backbone(x, alpha=alpha)

    # FPN
    p2 = create_fpn(features, fpn_ch=fpn_ch)

    # SimCC heads
    simcc_x, simcc_y = create_simcc_head(p2, num_bins=num_bins, simcc_ch=simcc_ch)

    # Decode coordinates
    coords = keras.layers.Lambda(
        lambda xy: decode_coords(xy[0], xy[1], tau=tau),
        name='coords'
    )([simcc_x, simcc_y])

    # Score head from c5
    c5 = features['c5']
    score = keras.layers.GlobalAveragePooling2D(name='score_gap')(c5)
    score_logit = keras.layers.Dense(1, name='score_logit')(score)

    model = keras.Model(
        inputs=inputs,
        outputs={
            'simcc_x': simcc_x,
            'simcc_y': simcc_y,
            'coords': coords,
            'score_logit': score_logit
        },
        name='DocCornerNetV3_QAT'
    )

    return model


# ============ Data Loading ============

def load_sample(image_path, label_path, img_size=224):
    """Load and preprocess a single sample."""
    # Load image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0

    # ImageNet normalization
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    img = (img - mean) / std

    return img


def create_dataset(data_root, split, batch_size, img_size=224, shuffle=True):
    """Create tf.data.Dataset for training/validation."""
    data_root = Path(data_root)

    # Load split file
    split_file = data_root / f"{split}.txt"
    with open(split_file) as f:
        filenames = [line.strip() for line in f if line.strip()]

    images_dir = data_root / "images"
    labels_dir = data_root / "labels"

    # Build lists
    image_paths = []
    coords_list = []
    has_doc_list = []

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
        has_doc_list.append(1.0)

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

            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std

            coords = np.array(coords_list[i], dtype=np.float32)
            has_doc = np.array([has_doc_list[i]], dtype=np.float32)

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


# ============ Loss ============

class SimCCLoss(keras.losses.Loss):
    def __init__(self, sigma_px=3.0, num_bins=224, **kwargs):
        super().__init__(**kwargs)
        self.sigma_px = sigma_px
        self.num_bins = num_bins

    def call(self, y_true, y_pred):
        # y_true: [B, 4] - normalized coords for one axis
        # y_pred: [B, 4, num_bins] - logits

        bins = tf.range(self.num_bins, dtype=tf.float32) / self.num_bins
        bins = tf.reshape(bins, [1, 1, -1])

        # Create Gaussian targets
        y_true_exp = tf.expand_dims(y_true, -1)  # [B, 4, 1]
        sigma_norm = self.sigma_px / self.num_bins
        targets = tf.exp(-0.5 * tf.square((bins - y_true_exp) / sigma_norm))
        targets = targets / (tf.reduce_sum(targets, axis=-1, keepdims=True) + 1e-8)

        # Cross-entropy loss
        pred_prob = tf.nn.softmax(y_pred, axis=-1)
        loss = -tf.reduce_sum(targets * tf.math.log(pred_prob + 1e-8), axis=-1)

        return tf.reduce_mean(loss)


def create_loss_fn(sigma_px=3.0, num_bins=224):
    simcc_loss = SimCCLoss(sigma_px=sigma_px, num_bins=num_bins)

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

        # Score BCE loss
        loss_score = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_has_doc, logits=score_logit)
        )

        return loss_simcc + 0.5 * loss_coord + 0.5 * loss_score

    return loss_fn


# ============ Main ============

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("QAT Training (Docker)")
    print("=" * 60)
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU: {tf.config.list_physical_devices('GPU')}")

    # Load config
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config_path = Path(args.saved_model).parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {'alpha': 0.75, 'fpn_ch': 48, 'simcc_ch': 128, 'num_bins': 224, 'img_size': 224, 'tau': 1.0}

    print(f"Config: {config}")

    # Load model from SavedModel
    print(f"\nLoading model from SavedModel: {args.saved_model}")
    model = tf.keras.models.load_model(args.saved_model, compile=False)
    print(f"Model loaded: {model.count_params():,} params")

    # Apply QAT
    print("\nApplying QAT...")
    import tensorflow_model_optimization as tfmot
    qat_model = tfmot.quantization.keras.quantize_model(model)
    print(f"QAT model params: {qat_model.count_params():,}")

    # Create datasets
    print("\nLoading datasets...")
    train_ds = create_dataset(args.data_root, 'train', args.batch_size, shuffle=True)
    val_ds = create_dataset(args.data_root, 'val', args.batch_size, shuffle=False)

    # Compile
    loss_fn = create_loss_fn(sigma_px=3.0, num_bins=config.get('num_bins', 224))
    qat_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=loss_fn
    )

    # Train
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

        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            qat_model.save(str(output_dir / "qat_best.keras"))
            print(f"  Saved best model")

    # Save final
    qat_model.save(str(output_dir / "qat_final.keras"))

    # Export to TFLite int8
    print("\nExporting to TFLite int8...")

    def representative_dataset():
        for images, _ in val_ds.take(50):
            for i in range(images.shape[0]):
                yield [images[i:i+1].numpy()]

    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    try:
        tflite_model = converter.convert()
        tflite_path = output_dir / "model_qat_int8.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"Saved: {tflite_path} ({len(tflite_model)/(1024*1024):.2f} MB)")
    except Exception as e:
        print(f"TFLite conversion error: {e}")
        print("Trying with SELECT_TF_OPS...")
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_model = converter.convert()
        tflite_path = output_dir / "model_qat_int8.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"Saved: {tflite_path} ({len(tflite_model)/(1024*1024):.2f} MB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
