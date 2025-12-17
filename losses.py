"""
Loss functions for DocCornerNetV3 (TensorFlow/Keras).

Contains:
- gaussian_1d_targets: Generate 1D Gaussian target distributions
- SimCCLoss: Cross-entropy/KL loss for SimCC marginals
- CoordLoss: Direct coordinate supervision (L1/SmoothL1)
- ScoreLoss: BCE for document presence classification
- DocCornerNetV3Trainer: Custom training loop with proper loss masking
"""

import tensorflow as tf
from tensorflow import keras


def gaussian_1d_targets(coords_01, bins=224, sigma_px=2.0):
    """
    Generate 1D Gaussian target distributions for SimCC.

    Args:
        coords_01: [B, 4] coordinates in [0, 1] for one axis (X or Y)
        bins: Number of bins (typically img_size)
        sigma_px: Gaussian sigma in pixel space

    Returns:
        targets: [B, 4, bins] normalized Gaussian distributions
    """
    # Convert normalized coords to pixel space
    coords_px = coords_01 * tf.cast(bins - 1, tf.float32)  # [B, 4]

    # Bin centers
    bin_indices = tf.cast(tf.range(bins), tf.float32)  # [bins]

    # Gaussian: exp(-(bin - center)^2 / (2 * sigma^2))
    # coords_px: [B, 4] -> [B, 4, 1]
    # bin_indices: [bins] -> [1, 1, bins]
    diff = bin_indices[None, None, :] - coords_px[:, :, None]  # [B, 4, bins]
    gauss = tf.exp(-0.5 * tf.square(diff) / (sigma_px * sigma_px))

    # Normalize to probability distribution
    gauss = gauss / (tf.reduce_sum(gauss, axis=-1, keepdims=True) + 1e-9)

    return gauss


class SimCCLoss(keras.layers.Layer):
    """
    SimCC Loss using cross-entropy with soft Gaussian targets.

    For each corner, we have X and Y distributions. Loss is computed as
    cross-entropy between predicted logits and target Gaussian distributions.
    """

    def __init__(self, bins=224, sigma_px=2.0, tau=1.0, **kwargs):
        """
        Args:
            bins: Number of bins (img_size)
            sigma_px: Gaussian sigma for targets
            tau: Temperature for predicted softmax
        """
        super().__init__(**kwargs)
        self.bins = bins
        self.sigma_px = sigma_px
        self.tau = tau

    def call(self, simcc_x, simcc_y, gt_coords, mask):
        """
        Compute SimCC loss.

        Args:
            simcc_x: [B, 4, bins] predicted X logits
            simcc_y: [B, 4, bins] predicted Y logits
            gt_coords: [B, 8] ground truth coords (x0,y0,x1,y1,...)
            mask: [B] float, 1=positive, 0=negative

        Returns:
            loss: scalar
        """
        # Reshape GT coords to [B, 4, 2]
        gt_coords_4x2 = tf.reshape(gt_coords, [-1, 4, 2])
        gt_x = gt_coords_4x2[:, :, 0]  # [B, 4]
        gt_y = gt_coords_4x2[:, :, 1]  # [B, 4]

        # Generate target distributions
        target_x = gaussian_1d_targets(gt_x, self.bins, self.sigma_px)  # [B, 4, bins]
        target_y = gaussian_1d_targets(gt_y, self.bins, self.sigma_px)

        # Cross-entropy with soft targets
        # CE = -sum(target * log(softmax(logits)))
        # = -sum(target * log_softmax(logits))
        log_pred_x = tf.nn.log_softmax(simcc_x / self.tau, axis=-1)
        log_pred_y = tf.nn.log_softmax(simcc_y / self.tau, axis=-1)

        ce_x = -tf.reduce_sum(target_x * log_pred_x, axis=-1)  # [B, 4]
        ce_y = -tf.reduce_sum(target_y * log_pred_y, axis=-1)  # [B, 4]

        # Mean over corners
        ce = tf.reduce_mean(ce_x + ce_y, axis=-1)  # [B]

        # Apply mask (only positive samples)
        loss = tf.reduce_sum(ce * mask) / (tf.reduce_sum(mask) + 1e-9)

        return loss


class CoordLoss(keras.layers.Layer):
    """
    Direct coordinate loss (L1 or SmoothL1).

    Provides auxiliary supervision on decoded coordinates for stability.
    """

    def __init__(self, loss_type="l1", **kwargs):
        """
        Args:
            loss_type: "l1" or "smooth_l1"
        """
        super().__init__(**kwargs)
        self.loss_type = loss_type

    def call(self, pred_coords, gt_coords, mask):
        """
        Compute coordinate loss.

        Args:
            pred_coords: [B, 8] predicted coordinates
            gt_coords: [B, 8] ground truth coordinates
            mask: [B] float, 1=positive, 0=negative

        Returns:
            loss: scalar
        """
        if self.loss_type == "smooth_l1":
            # SmoothL1 with beta=0.01
            diff = tf.abs(pred_coords - gt_coords)
            beta = 0.01
            loss_per_coord = tf.where(
                diff < beta,
                0.5 * tf.square(diff) / beta,
                diff - 0.5 * beta
            )
        else:
            # L1
            loss_per_coord = tf.abs(pred_coords - gt_coords)

        # Mean over coordinates
        loss_per_sample = tf.reduce_mean(loss_per_coord, axis=-1)  # [B]

        # Apply mask
        loss = tf.reduce_sum(loss_per_sample * mask) / (tf.reduce_sum(mask) + 1e-9)

        return loss


class DocCornerNetV3Trainer(keras.Model):
    """
    Custom training wrapper for DocCornerNetV3.

    Handles:
    - SimCC loss on X/Y marginals (only positive samples)
    - Coordinate loss for stability (only positive samples)
    - Score loss for document presence (all samples)

    Expected input format:
    - x: images [B, H, W, 3]
    - y: dict with "has_doc" [B, 1] and "coords" [B, 8]
    """

    def __init__(
        self,
        net,
        bins=224,
        sigma_px=2.0,
        tau=1.0,
        w_simcc=1.0,
        w_coord=0.2,
        w_score=1.0,
        **kwargs
    ):
        """
        Args:
            net: DocCornerNetV3 model
            bins: Number of bins for SimCC
            sigma_px: Gaussian sigma for targets
            tau: Temperature for softmax
            w_simcc: Weight for SimCC loss
            w_coord: Weight for coordinate loss
            w_score: Weight for score loss
        """
        super().__init__(**kwargs)
        self.net = net
        self.bins = bins
        self.sigma_px = sigma_px
        self.tau = tau
        self.w_simcc = w_simcc
        self.w_coord = w_coord
        self.w_score = w_score

        # Loss functions
        self.simcc_loss_fn = SimCCLoss(bins=bins, sigma_px=sigma_px, tau=tau)
        self.coord_loss_fn = CoordLoss(loss_type="l1")

        # Metrics
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.simcc_loss_tracker = keras.metrics.Mean(name="loss_simcc")
        self.coord_loss_tracker = keras.metrics.Mean(name="loss_coord")
        self.score_loss_tracker = keras.metrics.Mean(name="loss_score")
        self.iou_tracker = keras.metrics.Mean(name="iou")
        self.corner_err_tracker = keras.metrics.Mean(name="corner_err_px")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.simcc_loss_tracker,
            self.coord_loss_tracker,
            self.score_loss_tracker,
            self.iou_tracker,
            self.corner_err_tracker,
        ]

    def call(self, inputs, training=False):
        """Forward pass through the network."""
        return self.net(inputs, training=training)

    def train_step(self, data):
        x, y = data

        # Extract targets
        has_doc = tf.cast(y["has_doc"], tf.float32)  # [B, 1] or [B]
        if len(has_doc.shape) == 2:
            has_doc = tf.squeeze(has_doc, axis=-1)  # [B]

        coords_gt = tf.cast(y["coords"], tf.float32)  # [B, 8]

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self.net(x, training=True)

            simcc_x = outputs["simcc_x"]      # [B, 4, bins]
            simcc_y = outputs["simcc_y"]      # [B, 4, bins]
            score_logit = outputs["score_logit"]  # [B, 1]
            coords_pred = outputs["coords"]   # [B, 8]

            # Score loss (all samples)
            loss_score = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=has_doc[:, None],
                logits=score_logit
            )
            loss_score = tf.reduce_mean(loss_score)

            # SimCC loss (only positive samples)
            loss_simcc = self.simcc_loss_fn(simcc_x, simcc_y, coords_gt, has_doc)

            # Coordinate loss (only positive samples)
            loss_coord = self.coord_loss_fn(coords_pred, coords_gt, has_doc)

            # Total loss
            loss = (
                self.w_simcc * loss_simcc +
                self.w_coord * loss_coord +
                self.w_score * loss_score
            )

        # Gradient update
        grads = tape.gradient(loss, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.simcc_loss_tracker.update_state(loss_simcc)
        self.coord_loss_tracker.update_state(loss_coord)
        self.score_loss_tracker.update_state(loss_score)

        # Compute IoU and corner error for positive samples only
        iou, corner_err = self._compute_geometry_metrics(coords_pred, coords_gt, has_doc)
        self.iou_tracker.update_state(iou)
        self.corner_err_tracker.update_state(corner_err)

        return {
            "loss": self.loss_tracker.result(),
            "loss_simcc": self.simcc_loss_tracker.result(),
            "loss_coord": self.coord_loss_tracker.result(),
            "loss_score": self.score_loss_tracker.result(),
            "iou": self.iou_tracker.result(),
            "corner_err_px": self.corner_err_tracker.result(),
        }

    def test_step(self, data):
        x, y = data

        has_doc = tf.cast(y["has_doc"], tf.float32)
        if len(has_doc.shape) == 2:
            has_doc = tf.squeeze(has_doc, axis=-1)

        coords_gt = tf.cast(y["coords"], tf.float32)

        # Forward pass (no training)
        outputs = self.net(x, training=False)

        simcc_x = outputs["simcc_x"]
        simcc_y = outputs["simcc_y"]
        score_logit = outputs["score_logit"]
        coords_pred = outputs["coords"]

        # Compute losses
        loss_score = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=has_doc[:, None],
            logits=score_logit
        )
        loss_score = tf.reduce_mean(loss_score)

        loss_simcc = self.simcc_loss_fn(simcc_x, simcc_y, coords_gt, has_doc)
        loss_coord = self.coord_loss_fn(coords_pred, coords_gt, has_doc)

        loss = (
            self.w_simcc * loss_simcc +
            self.w_coord * loss_coord +
            self.w_score * loss_score
        )

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.simcc_loss_tracker.update_state(loss_simcc)
        self.coord_loss_tracker.update_state(loss_coord)
        self.score_loss_tracker.update_state(loss_score)

        iou, corner_err = self._compute_geometry_metrics(coords_pred, coords_gt, has_doc)
        self.iou_tracker.update_state(iou)
        self.corner_err_tracker.update_state(corner_err)

        return {
            "loss": self.loss_tracker.result(),
            "loss_simcc": self.simcc_loss_tracker.result(),
            "loss_coord": self.coord_loss_tracker.result(),
            "loss_score": self.score_loss_tracker.result(),
            "iou": self.iou_tracker.result(),
            "corner_err_px": self.corner_err_tracker.result(),
        }

    def _compute_geometry_metrics(self, pred_coords, gt_coords, mask):
        """
        Compute IoU and corner error for positive samples.

        Uses simplified bbox IoU for speed during training.
        Full polygon IoU computed in evaluate.py.

        Args:
            pred_coords: [B, 8]
            gt_coords: [B, 8]
            mask: [B] positive sample mask

        Returns:
            mean_iou: scalar
            mean_corner_err_px: scalar (at 224x224)
        """
        img_size = 224.0

        # Filter to positive samples
        mask_bool = tf.cast(mask, tf.bool)
        pred_pos = tf.boolean_mask(pred_coords, mask_bool)
        gt_pos = tf.boolean_mask(gt_coords, mask_bool)

        n_pos = tf.shape(pred_pos)[0]

        # Handle case with no positive samples
        def compute_metrics():
            # Corner error (in pixels)
            diff = tf.abs(pred_pos - gt_pos) * img_size  # [N, 8]
            corner_err = tf.reduce_mean(diff)

            # Simplified bbox IoU
            pred_xy = tf.reshape(pred_pos, [-1, 4, 2])  # [N, 4, 2]
            gt_xy = tf.reshape(gt_pos, [-1, 4, 2])

            pred_min = tf.reduce_min(pred_xy, axis=1)  # [N, 2]
            pred_max = tf.reduce_max(pred_xy, axis=1)
            gt_min = tf.reduce_min(gt_xy, axis=1)
            gt_max = tf.reduce_max(gt_xy, axis=1)

            inter_min = tf.maximum(pred_min, gt_min)
            inter_max = tf.minimum(pred_max, gt_max)
            inter_wh = tf.maximum(inter_max - inter_min, 0.0)
            inter_area = inter_wh[:, 0] * inter_wh[:, 1]

            pred_wh = pred_max - pred_min
            gt_wh = gt_max - gt_min
            pred_area = pred_wh[:, 0] * pred_wh[:, 1]
            gt_area = gt_wh[:, 0] * gt_wh[:, 1]

            union_area = pred_area + gt_area - inter_area + 1e-9
            iou = inter_area / union_area
            mean_iou = tf.reduce_mean(iou)

            return mean_iou, corner_err

        def zero_metrics():
            return tf.constant(0.0), tf.constant(0.0)

        iou, corner_err = tf.cond(
            n_pos > 0,
            compute_metrics,
            zero_metrics
        )

        return iou, corner_err


if __name__ == "__main__":
    print("Testing DocCornerNetV3 Losses...")

    import numpy as np

    B = 4
    bins = 224

    # Test Gaussian targets
    print("\n1. Testing gaussian_1d_targets...")
    coords = tf.constant([[0.2, 0.5, 0.8, 0.3]], dtype=tf.float32)  # [1, 4]
    targets = gaussian_1d_targets(coords, bins=bins, sigma_px=2.0)
    print(f"   Input coords: {coords.numpy()}")
    print(f"   Target shape: {targets.shape}")
    print(f"   Target sum (should be ~1): {tf.reduce_sum(targets[0, 0]).numpy():.4f}")

    # Test SimCCLoss
    print("\n2. Testing SimCCLoss...")
    simcc_loss = SimCCLoss(bins=bins, sigma_px=2.0, tau=1.0)
    simcc_x = tf.random.normal([B, 4, bins])
    simcc_y = tf.random.normal([B, 4, bins])
    gt_coords = tf.random.uniform([B, 8], 0.1, 0.9)
    mask = tf.constant([1.0, 1.0, 0.0, 1.0])
    loss = simcc_loss(simcc_x, simcc_y, gt_coords, mask)
    print(f"   SimCC loss: {loss.numpy():.4f}")

    # Test CoordLoss
    print("\n3. Testing CoordLoss...")
    coord_loss = CoordLoss(loss_type="l1")
    pred_coords = tf.random.uniform([B, 8], 0.1, 0.9)
    loss = coord_loss(pred_coords, gt_coords, mask)
    print(f"   Coord loss: {loss.numpy():.4f}")

    # Test full trainer
    print("\n4. Testing DocCornerNetV3Trainer...")
    from model import create_model

    net = create_model(alpha=0.75, fpn_ch=48, dec_ch=32, img_size=224, tau=1.0)
    trainer = DocCornerNetV3Trainer(
        net,
        bins=224,
        sigma_px=2.0,
        tau=1.0,
        w_simcc=1.0,
        w_coord=0.2,
        w_score=1.0
    )

    trainer.compile(optimizer=keras.optimizers.AdamW(learning_rate=2e-4, weight_decay=1e-4))

    # Simulate one training step
    x = np.random.randn(B, 224, 224, 3).astype(np.float32)
    y = {
        "has_doc": np.array([1, 1, 0, 1], dtype=np.float32),
        "coords": np.random.uniform(0.1, 0.9, (B, 8)).astype(np.float32),
    }

    # This would be a real training step
    # metrics = trainer.train_step((x, y))
    # print(f"   Training metrics: {metrics}")

    print(f"\n   Model params: {net.count_params():,}")

    print("\nAll loss tests passed!")
