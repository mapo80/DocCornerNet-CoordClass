"""
Quantization-Aware Training (QAT) for DocCornerNetV3.

Fine-tunes a pre-trained model with fake quantization layers to improve
int8 quantization accuracy.

Usage:
    python train_qat.py \
        --model checkpoints/best_model.keras \
        --data_root /path/to/dataset \
        --epochs 20
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import from existing modules
from dataset import DocCornerDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="QAT fine-tuning for DocCornerNetV3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset root")
    parser.add_argument("--train_file", type=str, default="train.txt",
                        help="Training split file")
    parser.add_argument("--val_file", type=str, default="val.txt",
                        help="Validation split file")
    parser.add_argument("--negative_dir", type=str, default="images-negative",
                        help="Negative samples directory")

    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="Path to pre-trained model (.keras)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.json (auto-detected if not specified)")

    # QAT Training
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of QAT epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (lower for fine-tuning)")

    # Loss weights (same as original)
    parser.add_argument("--sigma_px", type=float, default=3.0,
                        help="Gaussian sigma for SimCC targets")
    parser.add_argument("--w_simcc", type=float, default=1.0,
                        help="SimCC loss weight")
    parser.add_argument("--w_coord", type=float, default=0.5,
                        help="Coordinate loss weight")
    parser.add_argument("--w_score", type=float, default=0.5,
                        help="Score loss weight")

    # Output
    parser.add_argument("--output_dir", type=str, default="./checkpoints_qat",
                        help="Output directory for QAT models")

    return parser.parse_args()


def apply_qat(model):
    """
    Apply quantization-aware training to a model.

    This wraps the model with fake quantization nodes that simulate
    int8 quantization during training.
    """
    import tensorflow_model_optimization as tfmot

    # Define which layers to quantize
    # We'll use the default quantization config for most layers
    quantize_model = tfmot.quantization.keras.quantize_model

    # Apply QAT to the entire model
    qat_model = quantize_model(model)

    return qat_model


def apply_selective_qat(model):
    """
    Apply QAT selectively, keeping sensitive layers in float32.

    This is more conservative and may preserve accuracy better.
    """
    import tensorflow_model_optimization as tfmot
    from tensorflow_model_optimization.quantization.keras import quantize_annotate_layer
    from tensorflow_model_optimization.quantization.keras import quantize_apply

    # Clone model and annotate layers for quantization
    def apply_quantization_to_layer(layer):
        # Skip Lambda layers (soft-argmax, etc.) - they don't quantize well
        if isinstance(layer, keras.layers.Lambda):
            return layer
        # Skip final output layers to preserve coordinate precision
        if 'coords' in layer.name or 'score' in layer.name:
            return layer
        # Quantize everything else
        return quantize_annotate_layer(layer)

    # Clone with annotations
    annotated_model = keras.models.clone_model(
        model,
        clone_function=apply_quantization_to_layer,
    )

    # Apply quantization
    qat_model = quantize_apply(annotated_model)

    return qat_model


class SimCCLoss(keras.losses.Loss):
    """SimCC loss: KL divergence between predicted and target distributions."""

    def __init__(self, sigma_px=3.0, num_bins=224, **kwargs):
        super().__init__(**kwargs)
        self.sigma_px = sigma_px
        self.num_bins = num_bins
        # Pre-compute bin centers
        self.bin_centers = tf.cast(tf.range(num_bins), tf.float32) / num_bins

    def _create_gaussian_target(self, coord, sigma):
        """Create Gaussian target distribution centered at coord."""
        # coord: [B] normalized coordinate
        # Returns: [B, num_bins] probability distribution
        coord = tf.expand_dims(coord, -1)  # [B, 1]
        sigma_norm = sigma / self.num_bins
        diff = self.bin_centers - coord  # [B, num_bins]
        target = tf.exp(-0.5 * tf.square(diff / sigma_norm))
        target = target / (tf.reduce_sum(target, axis=-1, keepdims=True) + 1e-8)
        return target

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: [B, 4] normalized coordinates for 4 corners (one axis)
            y_pred: [B, 4, num_bins] predicted logits
        """
        # Apply softmax to predictions
        pred_prob = tf.nn.softmax(y_pred, axis=-1)  # [B, 4, num_bins]

        total_loss = 0.0
        for i in range(4):
            target = self._create_gaussian_target(y_true[:, i], self.sigma_px)
            # KL divergence: sum(target * log(target / pred))
            # Simplified as cross-entropy since target is fixed
            loss_i = -tf.reduce_sum(target * tf.math.log(pred_prob[:, i, :] + 1e-8), axis=-1)
            total_loss += loss_i

        return tf.reduce_mean(total_loss) / 4.0


def create_qat_loss_fn(sigma_px, num_bins, w_simcc, w_coord, w_score):
    """Create combined loss function for QAT training."""

    simcc_loss_x = SimCCLoss(sigma_px=sigma_px, num_bins=num_bins, name="simcc_x_loss")
    simcc_loss_y = SimCCLoss(sigma_px=sigma_px, num_bins=num_bins, name="simcc_y_loss")

    def loss_fn(y_true, y_pred):
        """
        Combined loss for QAT model.

        y_true: dict with 'coords' [B, 8] and 'has_document' [B, 1]
        y_pred: dict with 'simcc_x', 'simcc_y', 'score_logit', 'coords'
        """
        # Unpack ground truth
        gt_coords = y_true['coords']  # [B, 8]
        gt_has_doc = y_true['has_document']  # [B, 1]

        # Unpack predictions
        simcc_x = y_pred['simcc_x']  # [B, 4, num_bins]
        simcc_y = y_pred['simcc_y']  # [B, 4, num_bins]
        score_logit = y_pred['score_logit']  # [B, 1]
        pred_coords = y_pred['coords']  # [B, 8]

        # Extract X and Y coordinates
        gt_x = tf.stack([gt_coords[:, 0], gt_coords[:, 2], gt_coords[:, 4], gt_coords[:, 6]], axis=1)
        gt_y = tf.stack([gt_coords[:, 1], gt_coords[:, 3], gt_coords[:, 5], gt_coords[:, 7]], axis=1)

        # SimCC losses (only for positive samples)
        mask = tf.squeeze(gt_has_doc, axis=-1)  # [B]
        mask_expanded = tf.expand_dims(mask, -1)  # [B, 1]

        loss_simcc_x = simcc_loss_x(gt_x, simcc_x)
        loss_simcc_y = simcc_loss_y(gt_y, simcc_y)
        loss_simcc = (loss_simcc_x + loss_simcc_y) / 2.0

        # Coordinate loss (L1)
        coord_diff = tf.abs(pred_coords - gt_coords) * mask_expanded
        loss_coord = tf.reduce_sum(coord_diff) / (tf.reduce_sum(mask) * 8.0 + 1e-8)

        # Score loss (BCE)
        loss_score = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=gt_has_doc,
            logits=score_logit
        )
        loss_score = tf.reduce_mean(loss_score)

        # Combined loss
        total_loss = w_simcc * loss_simcc + w_coord * loss_coord + w_score * loss_score

        return total_loss

    return loss_fn


class QATTrainer:
    """Trainer for QAT fine-tuning."""

    def __init__(self, model, loss_fn, optimizer, train_dataset, val_dataset):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Metrics
        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.val_loss = keras.metrics.Mean(name='val_loss')
        self.val_iou = keras.metrics.Mean(name='val_iou')

    @tf.function
    def train_step(self, batch):
        images, targets = batch

        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.loss_fn(targets, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss.update_state(loss)
        return loss

    @tf.function
    def val_step(self, batch):
        images, targets = batch
        predictions = self.model(images, training=False)
        loss = self.loss_fn(targets, predictions)
        self.val_loss.update_state(loss)

        # Compute IoU for positive samples
        gt_coords = targets['coords']
        pred_coords = predictions['coords']
        mask = tf.squeeze(targets['has_document'], axis=-1)

        # Simple IoU approximation using coordinate distance
        coord_error = tf.reduce_mean(tf.abs(pred_coords - gt_coords), axis=-1)
        iou_approx = tf.maximum(0.0, 1.0 - coord_error * 10)  # Rough approximation
        iou_masked = tf.reduce_sum(iou_approx * mask) / (tf.reduce_sum(mask) + 1e-8)

        self.val_iou.update_state(iou_masked)

        return loss

    def fit(self, epochs, callbacks=None):
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'val_iou': []}

        for epoch in range(epochs):
            # Reset metrics
            self.train_loss.reset_state()
            self.val_loss.reset_state()
            self.val_iou.reset_state()

            # Training
            print(f"\nEpoch {epoch + 1}/{epochs}")
            for batch_idx, batch in enumerate(self.train_dataset):
                loss = self.train_step(batch)
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Batch {batch_idx + 1}: loss={self.train_loss.result():.4f}")

            # Validation
            for batch in self.val_dataset:
                self.val_step(batch)

            train_loss = self.train_loss.result().numpy()
            val_loss = self.val_loss.result().numpy()
            val_iou = self.val_iou.result().numpy()

            history['train_loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))
            history['val_iou'].append(float(val_iou))

            print(f"  train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_iou={val_iou:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if callbacks:
                    for cb in callbacks:
                        if hasattr(cb, 'on_best_model'):
                            cb.on_best_model(self.model, epoch, val_loss)

        return history


def create_tf_dataset(dataset, batch_size, shuffle=True):
    """Convert DocCornerDataset to tf.data.Dataset."""

    def generator():
        indices = list(range(len(dataset)))
        if shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            sample = dataset[idx]
            image = sample['image']
            coords = sample['coords']
            has_doc = sample['has_document']

            yield (
                image,
                {
                    'coords': coords,
                    'has_document': np.array([has_doc], dtype=np.float32)
                }
            )

    output_signature = (
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
        {
            'coords': tf.TensorSpec(shape=(8,), dtype=tf.float32),
            'has_document': tf.TensorSpec(shape=(1,), dtype=tf.float32)
        }
    )

    tf_dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    tf_dataset = tf_dataset.batch(batch_size)
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

    return tf_dataset


def main():
    args = parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("QAT Training for DocCornerNetV3")
    print("=" * 60)

    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(args.model).parent / "config.json"

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
    else:
        config = {
            'alpha': 0.75,
            'fpn_ch': 48,
            'simcc_ch': 128,
            'img_size': 224,
            'num_bins': 224,
            'tau': 1.0,
        }
        print("Using default config")

    # Load full .keras model directly (preserves all weights correctly)
    print(f"\nLoading model from {args.model}...")
    model = keras.models.load_model(args.model, compile=False)
    print(f"Model parameters: {model.count_params():,}")

    # Apply QAT
    print("\nApplying Quantization-Aware Training...")
    try:
        import tensorflow_model_optimization as tfmot
        print(f"tensorflow_model_optimization version: {tfmot.__version__}")

        # Use full model quantization
        qat_model = tfmot.quantization.keras.quantize_model(model)
        print(f"QAT model parameters: {qat_model.count_params():,}")

    except ImportError:
        print("ERROR: tensorflow_model_optimization not installed!")
        print("Install with: pip install tensorflow-model-optimization")
        return

    # Create datasets
    print("\nLoading datasets...")
    data_root = Path(args.data_root)

    # Load train samples
    train_file = data_root / args.train_file
    with open(train_file) as f:
        train_samples = [line.strip() for line in f if line.strip()]

    # Load val samples
    val_file = data_root / args.val_file
    with open(val_file) as f:
        val_samples = [line.strip() for line in f if line.strip()]

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    # Create dataset objects
    augment_config = {
        "rotation_degrees": 5,
        "scale_range": (0.9, 1.05),
        "brightness": 0.15,
        "contrast": 0.15,
        "saturation": 0.1,
        "blur_prob": 0.1,
        "blur_kernel": 3,
        "horizontal_flip": True,
    }

    train_dataset = DocCornerDataset(
        samples=train_samples,
        image_dir=data_root / "images",
        labels_dir=data_root / "labels",
        img_size=config.get('img_size', 224),
        augment=True,
        augment_config=augment_config,
    )

    val_dataset = DocCornerDataset(
        samples=val_samples,
        image_dir=data_root / "images",
        labels_dir=data_root / "labels",
        img_size=config.get('img_size', 224),
        augment=False,
    )

    # Convert to tf.data
    train_tf = create_tf_dataset(train_dataset, args.batch_size, shuffle=True)
    val_tf = create_tf_dataset(val_dataset, args.batch_size, shuffle=False)

    # Create loss function
    loss_fn = create_qat_loss_fn(
        sigma_px=args.sigma_px,
        num_bins=config.get('num_bins', 224),
        w_simcc=args.w_simcc,
        w_coord=args.w_coord,
        w_score=args.w_score,
    )

    # Create optimizer with lower learning rate for fine-tuning
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    # Compile QAT model
    qat_model.compile(optimizer=optimizer, loss=loss_fn)

    # Training callbacks
    class SaveBestCallback:
        def __init__(self, output_dir):
            self.output_dir = output_dir

        def on_best_model(self, model, epoch, val_loss):
            # Save full QAT model
            model_path = self.output_dir / "qat_best_model.keras"
            model.save(str(model_path))
            print(f"  Saved best model (epoch {epoch + 1}, val_loss={val_loss:.4f})")

    # Create trainer
    trainer = QATTrainer(
        model=qat_model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_dataset=train_tf,
        val_dataset=val_tf,
    )

    # Train
    print(f"\nStarting QAT training for {args.epochs} epochs...")
    history = trainer.fit(
        epochs=args.epochs,
        callbacks=[SaveBestCallback(output_dir)]
    )

    # Save final model (both full model and weights)
    print("\nSaving final QAT model...")
    qat_model.save(str(output_dir / "qat_final_model.keras"))
    qat_model.save_weights(str(output_dir / "qat_final_model.weights.h5"))

    # Save training history
    with open(output_dir / "qat_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Export to TFLite int8
    print("\nExporting to TFLite int8...")

    # Create representative dataset
    def representative_dataset():
        for i, (images, _) in enumerate(val_tf.take(100)):
            for j in range(images.shape[0]):
                yield [images[j:j+1].numpy()]

    # Convert QAT model to TFLite
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

        print(f"Saved: {tflite_path} ({len(tflite_model) / (1024*1024):.2f} MB)")

    except Exception as e:
        print(f"TFLite conversion failed: {e}")
        print("Trying with SELECT_TF_OPS fallback...")

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]

        tflite_model = converter.convert()

        tflite_path = output_dir / "model_qat_int8.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        print(f"Saved: {tflite_path} ({len(tflite_model) / (1024*1024):.2f} MB)")

    print("\n" + "=" * 60)
    print("QAT Training Complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
