"""
DocCornerNetV3: Document Corner Detection with SimCC (TensorFlow/Keras).

Architecture:
- Backbone: MobileNetV3Small (alpha=0.75) with ImageNet weights
- Neck: Mini-FPN merging C2 (56x56), C3 (28x28), C4 (14x14)
- Head: SimCC - direct 1D coordinate classification (MMPose style)
- Output: 4 corner coordinates + document presence score

SimCC (Simple Coordinate Classification) - MMPose style:
- Predicts X and Y coordinates as 1D classification problems
- Features are processed through FC layers to produce logits directly
- NO 2D heatmap intermediate - direct 1D prediction
- Soft-argmax decodes logits to continuous coordinates
- More efficient than 2D heatmaps, maintains sub-pixel precision

Target: <1M parameters, IoU >= 0.99 at 224x224
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _get_feature_layers(backbone):
    """
    Extract multi-scale feature layers from MobileNetV3Small backbone.

    Returns outputs at different scales:
    - C2: 56x56, C3: 28x28, C4: 14x14, C5: 7x7
    """
    c2 = c3 = c4 = c5 = None

    for layer in backbone.layers:
        out = layer.output
        if not hasattr(out, 'shape') or len(out.shape) != 4:
            continue

        _, h, w, c = out.shape
        if h == 1 and w == 1:
            continue

        if h == 56 and w == 56:
            c2 = out
        elif h == 28 and w == 28:
            c3 = out
        elif h == 14 and w == 14:
            c4 = out
        elif h == 7 and w == 7:
            c5 = out

    if c2 is None or c3 is None or c4 is None or c5 is None:
        raise ValueError("Could not find all feature scales in backbone")

    return c2, c3, c4, c5


def _separable_conv_block(x, filters, name):
    """Depthwise separable conv block: SepConv -> BN -> Swish"""
    x = layers.SeparableConv2D(
        filters, 3, padding="same", use_bias=False, name=f"{name}_sepconv"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("swish", name=f"{name}_swish")(x)
    return x


def build_doccorner_simcc_v3(
    alpha: float = 0.75,
    fpn_ch: int = 48,
    simcc_ch: int = 128,
    img_size: int = 224,
    num_bins: int = 224,
    tau: float = 1.0,
    score_init_bias: float = 1.75,
    backbone_weights="imagenet",
):
    """
    Build DocCornerNetV3 model with correct SimCC (MMPose style).

    SimCC predicts X and Y coordinates directly as 1D classification:
    - Features → FC → simcc_x [B, 4, num_bins]
    - Features → FC → simcc_y [B, 4, num_bins]
    - Soft-argmax → coords [B, 8]

    Args:
        alpha: MobileNetV3 width multiplier (0.75 or 1.0)
        fpn_ch: FPN intermediate channels
        simcc_ch: SimCC head hidden channels
        img_size: Input image size (default 224)
        num_bins: Number of bins for coordinate classification
        tau: Temperature for softmax in coordinate decode
        score_init_bias: Initial bias for score head

    Returns:
        Keras Model with outputs dict
    """
    inp = keras.Input((img_size, img_size, 3), name="image")

    # =========================================================================
    # Backbone: MobileNetV3Small (optionally ImageNet pretrained)
    # =========================================================================
    backbone = keras.applications.MobileNetV3Small(
        input_tensor=inp,
        include_top=False,
        weights=backbone_weights,
        alpha=alpha,
        include_preprocessing=True,
    )

    c2, c3, c4, c5 = _get_feature_layers(backbone)

    # =========================================================================
    # Mini-FPN: Merge multi-scale features
    # =========================================================================
    p4 = layers.Conv2D(fpn_ch, 1, padding="same", use_bias=False, name="fpn_lat_c4")(c4)
    p4 = layers.BatchNormalization(name="fpn_lat_c4_bn")(p4)

    p3 = layers.Conv2D(fpn_ch, 1, padding="same", use_bias=False, name="fpn_lat_c3")(c3)
    p3 = layers.BatchNormalization(name="fpn_lat_c3_bn")(p3)

    p2 = layers.Conv2D(fpn_ch, 1, padding="same", use_bias=False, name="fpn_lat_c2")(c2)
    p2 = layers.BatchNormalization(name="fpn_lat_c2_bn")(p2)

    # Top-down pathway
    p4_up = layers.UpSampling2D(size=2, interpolation="nearest", name="fpn_p4_up")(p4)
    p3 = layers.Add(name="fpn_p3_add")([p3, p4_up])
    p3 = _separable_conv_block(p3, fpn_ch, "fpn_p3_refine")

    p3_up = layers.UpSampling2D(size=2, interpolation="nearest", name="fpn_p3_up")(p3)
    p2 = layers.Add(name="fpn_p2_add")([p2, p3_up])
    p2 = _separable_conv_block(p2, fpn_ch, "fpn_p2_refine")  # [B, 56, 56, fpn_ch]

    # =========================================================================
    # SimCC Head: Spatial-aware coordinate classification
    # =========================================================================
    # Instead of GAP which loses spatial info, use 1D convolutions along axes
    # This preserves spatial structure while reducing to 1D distributions

    # P2 is [B, 56, 56, fpn_ch] - high resolution features
    # P3 is [B, 28, 28, fpn_ch] - medium resolution features

    # Multi-scale fusion: combine P2 and upsampled P3 for richer features
    p3_up_feat = layers.UpSampling2D(size=2, interpolation="bilinear", name="simcc_p3_up")(p3)
    p_fused = layers.Concatenate(name="simcc_fuse")([p2, p3_up_feat])  # [B, 56, 56, 2*fpn_ch]

    # Refine fused features
    p_fused = _separable_conv_block(p_fused, fpn_ch * 2, "simcc_refine1")
    p_fused = _separable_conv_block(p_fused, fpn_ch, "simcc_refine2")  # [B, 56, 56, fpn_ch]

    # Generate X marginals: reduce along Y axis (vertical pooling)
    # [B, 56, 56, fpn_ch] -> [B, 56, fpn_ch] via mean along axis 1
    x_marginal = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1),
        name="x_marginal_pool"
    )(p_fused)  # [B, 56, fpn_ch]

    # Generate Y marginals: reduce along X axis (horizontal pooling)
    # [B, 56, 56, fpn_ch] -> [B, 56, fpn_ch] via mean along axis 2
    y_marginal = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=2),
        name="y_marginal_pool"
    )(p_fused)  # [B, 56, fpn_ch]

    # Upsample marginals to num_bins resolution
    # [B, 56, fpn_ch] -> [B, num_bins, fpn_ch]
    x_marginal = layers.Lambda(
        lambda t, nb=num_bins: tf.image.resize(t[:, :, None, :], (nb, 1))[:, :, 0, :],
        name="x_marginal_resize"
    )(x_marginal)  # [B, 224, fpn_ch]

    y_marginal = layers.Lambda(
        lambda t, nb=num_bins: tf.image.resize(t[:, :, None, :], (nb, 1))[:, :, 0, :],
        name="y_marginal_resize"
    )(y_marginal)  # [B, 224, fpn_ch]

    # 1D convolutions for coordinate refinement along each axis
    x_feat = layers.Conv1D(simcc_ch, 5, padding="same", name="simcc_x_conv1")(x_marginal)
    x_feat = layers.BatchNormalization(name="simcc_x_bn1")(x_feat)
    x_feat = layers.ReLU(name="simcc_x_relu1")(x_feat)
    x_feat = layers.Conv1D(simcc_ch // 2, 3, padding="same", name="simcc_x_conv2")(x_feat)
    x_feat = layers.BatchNormalization(name="simcc_x_bn2")(x_feat)
    x_feat = layers.ReLU(name="simcc_x_relu2")(x_feat)

    y_feat = layers.Conv1D(simcc_ch, 5, padding="same", name="simcc_y_conv1")(y_marginal)
    y_feat = layers.BatchNormalization(name="simcc_y_bn1")(y_feat)
    y_feat = layers.ReLU(name="simcc_y_relu1")(y_feat)
    y_feat = layers.Conv1D(simcc_ch // 2, 3, padding="same", name="simcc_y_conv2")(y_feat)
    y_feat = layers.BatchNormalization(name="simcc_y_bn2")(y_feat)
    y_feat = layers.ReLU(name="simcc_y_relu2")(y_feat)

    # Global context via GAP (auxiliary, for corner disambiguation)
    global_feat = layers.GlobalAveragePooling2D(name="simcc_global_gap")(p_fused)  # [B, fpn_ch]
    global_feat = layers.Dense(simcc_ch // 2, name="simcc_global_fc")(global_feat)
    global_feat = layers.ReLU(name="simcc_global_relu")(global_feat)

    # Broadcast global features to each position
    # [B, simcc_ch//2] -> [B, num_bins, simcc_ch//2]
    global_x = layers.Lambda(
        lambda t, nb=num_bins: tf.tile(t[:, None, :], [1, nb, 1]),
        name="global_x_broadcast"
    )(global_feat)
    global_y = layers.Lambda(
        lambda t, nb=num_bins: tf.tile(t[:, None, :], [1, nb, 1]),
        name="global_y_broadcast"
    )(global_feat)

    # Concatenate local and global features
    x_feat = layers.Concatenate(name="simcc_x_cat")([x_feat, global_x])  # [B, 224, simcc_ch]
    y_feat = layers.Concatenate(name="simcc_y_cat")([y_feat, global_y])  # [B, 224, simcc_ch]

    # Output heads: 4 corners for each axis
    # [B, num_bins, simcc_ch] -> [B, num_bins, 4]
    simcc_x = layers.Conv1D(
        4, 1,
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=keras.initializers.Zeros(),
        name="simcc_x_out"
    )(x_feat)  # [B, 224, 4]
    simcc_x = layers.Permute((2, 1), name="simcc_x")(simcc_x)  # [B, 4, 224]

    simcc_y = layers.Conv1D(
        4, 1,
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=keras.initializers.Zeros(),
        name="simcc_y_out"
    )(y_feat)  # [B, 224, 4]
    simcc_y = layers.Permute((2, 1), name="simcc_y")(simcc_y)  # [B, 4, 224]

    # =========================================================================
    # Coordinate Decode: Soft-argmax
    # =========================================================================
    def decode_coords(inputs):
        """Decode SimCC logits to normalized coordinates using soft-argmax."""
        sx, sy = inputs  # [B, 4, num_bins]

        sx = tf.cast(sx, tf.float32)
        sy = tf.cast(sy, tf.float32)

        bins = tf.shape(sx)[-1]
        centers = tf.linspace(0.0, 1.0, bins)

        # Softmax with temperature
        px = tf.nn.softmax(sx / tau, axis=-1)
        py = tf.nn.softmax(sy / tau, axis=-1)

        # Expected value (soft-argmax)
        x = tf.reduce_sum(px * centers[None, None, :], axis=-1)
        y = tf.reduce_sum(py * centers[None, None, :], axis=-1)

        # Interleave to [x0, y0, x1, y1, x2, y2, x3, y3]
        coords = tf.stack([
            x[:, 0], y[:, 0],
            x[:, 1], y[:, 1],
            x[:, 2], y[:, 2],
            x[:, 3], y[:, 3],
        ], axis=-1)

        return tf.clip_by_value(coords, 0.0, 1.0)

    coords = layers.Lambda(
        decode_coords,
        output_shape=(8,),
        name="coords"
    )([simcc_x, simcc_y])

    # =========================================================================
    # Score Head: Document presence classification
    # =========================================================================
    score_features = layers.GlobalAveragePooling2D(name="score_gap")(c5)
    score_logit = layers.Dense(
        1,
        bias_initializer=keras.initializers.Constant(score_init_bias),
        name="score_logit"
    )(score_features)

    # =========================================================================
    # Build Model
    # =========================================================================
    model = keras.Model(
        inputs=inp,
        outputs={
            "simcc_x": simcc_x,
            "simcc_y": simcc_y,
            "score_logit": score_logit,
            "coords": coords,
        },
        name="DocCornerNetV3_SimCC"
    )

    return model


def create_model(
    alpha: float = 0.75,
    fpn_ch: int = 48,
    simcc_ch: int = 128,
    img_size: int = 224,
    num_bins: int = 224,
    tau: float = 1.0,
    score_init_bias: float = 1.75,
    backbone_weights="imagenet",
):
    """
    Factory function to create DocCornerNetV3 training model with SimCC.

    Returns model with dict outputs for training:
    - simcc_x: [B, 4, num_bins] - X coordinate logits
    - simcc_y: [B, 4, num_bins] - Y coordinate logits
    - score_logit: [B, 1] - document presence logit
    - coords: [B, 8] - decoded coordinates

    Args:
        alpha: MobileNetV3 width multiplier (default 0.75)
        fpn_ch: FPN channels (default 48)
        simcc_ch: SimCC head channels (default 128)
        img_size: Input size (default 224)
        num_bins: Coordinate bins (default 224)
        tau: Softmax temperature (default 1.0)
        score_init_bias: Score bias init (default 1.75)

    Returns:
        DocCornerNetV3 Keras Model (training version)
    """
    return build_doccorner_simcc_v3(
        alpha=alpha,
        fpn_ch=fpn_ch,
        simcc_ch=simcc_ch,
        img_size=img_size,
        num_bins=num_bins,
        tau=tau,
        score_init_bias=score_init_bias,
        backbone_weights=backbone_weights,
    )


def create_inference_model(train_model):
    """
    Create inference model from training model with simplified output.

    Output format matches DocCornerNet-MobileNetV3-Enhanced:
    - coords: [B, 8] - normalized corner coordinates (x0,y0,x1,y1,x2,y2,x3,y3)
    - score: [B, 1] - document presence logit (apply sigmoid for probability)

    The inference model shares weights with the training model.

    Args:
        train_model: Training model created by create_model()

    Returns:
        Keras Model with tuple output (coords, score)
    """
    # Extract the relevant outputs from training model
    coords = train_model.output["coords"]
    score_logit = train_model.output["score_logit"]

    # Create inference model with simplified outputs
    inference_model = keras.Model(
        inputs=train_model.input,
        outputs=[coords, score_logit],
        name="DocCornerNetV3_Inference"
    )

    return inference_model


def load_inference_model(weights_path: str, **model_kwargs):
    """
    Load inference model from saved weights.

    Args:
        weights_path: Path to .weights.h5 file
        **model_kwargs: Arguments for create_model (alpha, fpn_ch, etc.)

    Returns:
        Inference model with weights loaded
    """
    # Create training model to load weights
    train_model = create_model(**model_kwargs)
    train_model.load_weights(weights_path)

    # Convert to inference model
    return create_inference_model(train_model)


if __name__ == "__main__":
    import numpy as np

    print("Testing DocCornerNetV3 with SimCC (TensorFlow)...")
    print("=" * 60)

    # Create training model
    train_model = create_model(
        alpha=0.75,
        fpn_ch=48,
        simcc_ch=128,
        img_size=224,
        num_bins=224,
        tau=1.0,
    )

    print(f"\n[Training Model]")
    print(f"Name: {train_model.name}")
    print(f"Parameters: {train_model.count_params():,}")

    if train_model.count_params() < 1_000_000:
        print("✓ Under 1M parameters target")
    else:
        print(f"✗ Over 1M parameters by {train_model.count_params() - 1_000_000:,}")

    # Test training model forward pass
    print("\nTesting training model forward pass...")
    x = np.random.randn(2, 224, 224, 3).astype(np.float32)
    outputs = train_model(x, training=False)

    print(f"  simcc_x: {outputs['simcc_x'].shape}")
    print(f"  simcc_y: {outputs['simcc_y'].shape}")
    print(f"  score_logit: {outputs['score_logit'].shape}")
    print(f"  coords: {outputs['coords'].shape}")

    # Create inference model
    print("\n" + "-" * 60)
    print("[Inference Model]")
    inference_model = create_inference_model(train_model)
    print(f"Name: {inference_model.name}")
    print(f"Parameters: {inference_model.count_params():,}")

    # Test inference model forward pass
    print("\nTesting inference model forward pass...")
    coords, score_logit = inference_model(x, training=False)

    print(f"  coords: {coords.shape}")
    print(f"  score_logit: {score_logit.shape}")

    # Verify outputs match
    print("\nVerifying outputs match between models...")
    train_coords = outputs['coords'].numpy()
    infer_coords = coords.numpy()
    coords_match = np.allclose(train_coords, infer_coords)
    print(f"  Coords match: {'✓' if coords_match else '✗'}")

    train_score = outputs['score_logit'].numpy()
    infer_score = score_logit.numpy()
    score_match = np.allclose(train_score, infer_score)
    print(f"  Score match: {'✓' if score_match else '✗'}")

    # Show sample output
    print(f"\nCoords (sample 0): {infer_coords[0]}")
    print(f"Coords range: [{infer_coords.min():.4f}, {infer_coords.max():.4f}]")

    score_prob = tf.nn.sigmoid(score_logit).numpy()
    print(f"Score (sigmoid): {score_prob.flatten()}")

    print("\n" + "=" * 60)
    if coords_match and score_match:
        print("All tests passed!")
    else:
        print("TESTS FAILED!")
