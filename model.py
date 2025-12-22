"""
DocCornerNetV3: Document Corner Detection with SimCC (TensorFlow/Keras).

Architecture:
- Backbone: MobileNetV3Small/Large (alpha width multiplier, optional minimalistic variant)
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
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="doccorner")
class AxisMean(layers.Layer):
    def __init__(self, axis: int, impl: str = "mean", **kwargs):
        super().__init__(**kwargs)
        self.axis = int(axis)
        self.impl = str(impl).lower().strip()
        self._h = None
        self._w = None
        self._c = None
        self._filter_full = None
        self._filter_2 = None
        self._filter_rem = None
        self._filters_strided = None
        self._strides_strided = None
        self._steps = 0
        self._rem = 1

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(f"AxisMean expects rank-4 NHWC input, got shape={input_shape}")
        h, w, c = input_shape[1], input_shape[2], input_shape[3]
        if h is None or w is None or c is None:
            raise ValueError("AxisMean requires static H/W/C for TFLite/XNNPACK-friendly export.")
        self._h = int(h)
        self._w = int(w)
        self._c = int(c)

        impl = self.impl
        if impl == "mean":
            super().build(input_shape)
            return

        if self.axis not in (1, 2):
            raise ValueError(f"AxisMean supports only axis=1 or axis=2, got axis={self.axis}")

        if impl == "avgpool":
            super().build(input_shape)
            return

        if impl == "dwconv_full":
            if self.axis == 1:
                k = np.ones((self._h, 1, self._c, 1), dtype=np.float32) / float(self._h)
            else:
                k = np.ones((1, self._w, self._c, 1), dtype=np.float32) / float(self._w)
            self._filter_full = tf.constant(k, dtype=tf.float32, name=f"{self.name}_dwfilter_full")
            super().build(input_shape)
            return

        if impl == "dwconv_strided":
            length = self._h if self.axis == 1 else self._w
            filters = []
            strides = []
            factors = (8, 4, 2)
            while length > 1:
                k = None
                for f in factors:
                    if length % f == 0:
                        k = f
                        break
                if k is None:
                    k = int(length)
                if self.axis == 1:
                    kk = np.ones((k, 1, self._c, 1), dtype=np.float32) / float(k)
                    filters.append(tf.constant(kk, dtype=tf.float32, name=f"{self.name}_dwfilter_k{k}"))
                    strides.append([1, k, 1, 1])
                else:
                    kk = np.ones((1, k, self._c, 1), dtype=np.float32) / float(k)
                    filters.append(tf.constant(kk, dtype=tf.float32, name=f"{self.name}_dwfilter_k{k}"))
                    strides.append([1, 1, k, 1])
                length //= k
            self._filters_strided = filters
            self._strides_strided = strides
            super().build(input_shape)
            return

        if impl == "dwconv_pyramid":
            length = self._h if self.axis == 1 else self._w
            steps = 0
            while length > 1 and (length % 2 == 0):
                steps += 1
                length //= 2
            self._steps = int(steps)
            self._rem = int(length)

            if self.axis == 1:
                k2 = np.ones((2, 1, self._c, 1), dtype=np.float32) * 0.5
                self._filter_2 = tf.constant(k2, dtype=tf.float32, name=f"{self.name}_dwfilter2")
                if self._rem > 1:
                    krem = np.ones((self._rem, 1, self._c, 1), dtype=np.float32) / float(self._rem)
                    self._filter_rem = tf.constant(krem, dtype=tf.float32, name=f"{self.name}_dwfilter_rem")
            else:
                k2 = np.ones((1, 2, self._c, 1), dtype=np.float32) * 0.5
                self._filter_2 = tf.constant(k2, dtype=tf.float32, name=f"{self.name}_dwfilter2")
                if self._rem > 1:
                    krem = np.ones((1, self._rem, self._c, 1), dtype=np.float32) / float(self._rem)
                    self._filter_rem = tf.constant(krem, dtype=tf.float32, name=f"{self.name}_dwfilter_rem")
            super().build(input_shape)
            return

        raise ValueError(f"Unsupported AxisMean impl='{self.impl}'")
        super().build(input_shape)

    def call(self, inputs):
        if self.impl == "mean":
            return tf.reduce_mean(inputs, axis=self.axis)

        if self._h is None or self._w is None or self._c is None:
            raise RuntimeError("AxisMean is not built.")

        if self.impl == "avgpool":
            if self.axis == 1:
                x = tf.nn.avg_pool2d(inputs, ksize=(self._h, 1), strides=(1, 1), padding="VALID")
                return tf.reshape(x, [-1, self._w, self._c])
            x = tf.nn.avg_pool2d(inputs, ksize=(1, self._w), strides=(1, 1), padding="VALID")
            return tf.reshape(x, [-1, self._h, self._c])

        if self.impl == "dwconv_full":
            if self._filter_full is None:
                raise RuntimeError("AxisMean dwconv_full filter missing.")
            x = tf.nn.depthwise_conv2d(inputs, self._filter_full, strides=[1, 1, 1, 1], padding="VALID")
            if self.axis == 1:
                return tf.reshape(x, [-1, self._w, self._c])
            return tf.reshape(x, [-1, self._h, self._c])

        if self.impl == "dwconv_strided":
            if self._filters_strided is None or self._strides_strided is None:
                raise RuntimeError("AxisMean dwconv_strided filters missing.")
            x = inputs
            for f, s in zip(self._filters_strided, self._strides_strided, strict=False):
                x = tf.nn.depthwise_conv2d(x, f, strides=s, padding="VALID")
            if self.axis == 1:
                return tf.reshape(x, [-1, self._w, self._c])
            return tf.reshape(x, [-1, self._h, self._c])

        if self.impl == "dwconv_pyramid":
            if self._filter_2 is None:
                raise RuntimeError("AxisMean dwconv_pyramid filter2 missing.")
            x = inputs
            if self.axis == 1:
                for _ in range(self._steps):
                    x = tf.nn.depthwise_conv2d(x, self._filter_2, strides=[1, 2, 1, 1], padding="VALID")
                if self._rem > 1:
                    if self._filter_rem is None:
                        raise RuntimeError("AxisMean dwconv_pyramid filter_rem missing.")
                    x = tf.nn.depthwise_conv2d(x, self._filter_rem, strides=[1, 1, 1, 1], padding="VALID")
                return tf.reshape(x, [-1, self._w, self._c])

            for _ in range(self._steps):
                x = tf.nn.depthwise_conv2d(x, self._filter_2, strides=[1, 1, 2, 1], padding="VALID")
            if self._rem > 1:
                if self._filter_rem is None:
                    raise RuntimeError("AxisMean dwconv_pyramid filter_rem missing.")
                x = tf.nn.depthwise_conv2d(x, self._filter_rem, strides=[1, 1, 1, 1], padding="VALID")
            return tf.reshape(x, [-1, self._h, self._c])

        raise ValueError(f"Unsupported AxisMean impl='{self.impl}'")

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "impl": self.impl})
        return config


@register_keras_serializable(package="doccorner")
class Resize1D(layers.Layer):
    def __init__(self, target_length: int, method: str = "bilinear", **kwargs):
        super().__init__(**kwargs)
        self.target_length = int(target_length)
        self.method = str(method)

    def call(self, inputs):
        # inputs: [B, L, C] -> [B, target_length, C]
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError("Resize1D requires a known channel dimension for TFLite/XNNPACK-friendly export.")
        length = inputs.shape[1]
        if length is None:
            raise ValueError("Resize1D requires a known length dimension for TFLite/XNNPACK-friendly export.")
        # Avoid EXPAND_DIMS: reshape to 4D for RESIZE_BILINEAR.
        x = tf.reshape(inputs, [-1, int(length), 1, int(channels)])  # [B, L, 1, C]
        x = tf.image.resize(x, size=(self.target_length, 1), method=self.method)
        # Avoid STRIDED_SLICE (often not delegated): reshape away the singleton width dimension.
        return tf.reshape(x, [-1, self.target_length, int(channels)])

    def get_config(self):
        config = super().get_config()
        config.update({"target_length": self.target_length, "method": self.method})
        return config


@register_keras_serializable(package="doccorner")
class Broadcast1D(layers.Layer):
    def __init__(self, target_length: int, **kwargs):
        super().__init__(**kwargs)
        self.target_length = int(target_length)
        self._channels = None
        self._ones = None

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError("Broadcast1D requires a known channel dimension for TFLite/XNNPACK-friendly export.")
        self._channels = int(channels)
        # Constant ones used to broadcast from [B,1,C] -> [B,target_length,C] via MUL broadcasting.
        self._ones = tf.constant(
            np.ones((1, self.target_length, 1), dtype=np.float32),
            dtype=tf.float32,
            name=f"{self.name}_ones",
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: [B, C] -> [B, target_length, C]
        if self._channels is None or self._ones is None:
            raise RuntimeError("Broadcast1D is not built.")
        x = tf.reshape(inputs, [-1, 1, self._channels])
        # Broadcast along the length dimension without TILE.
        return x * self._ones

    def get_config(self):
        config = super().get_config()
        config.update({"target_length": self.target_length})
        return config


@register_keras_serializable(package="doccorner")
class Conv1DAsConv2D(layers.Layer):
    """
    XNNPACK-friendly Conv1D equivalent implemented as:
      [B,L,C] -> RESHAPE -> [B,1,L,C] -> CONV_2D -> RESHAPE -> [B,L,F]

    This avoids EXPAND_DIMS/SQUEEZE patterns that prevent full delegation for int8 graphs.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        padding: str = "same",
        use_bias: bool = True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.strides = int(strides)
        self.padding = str(padding).lower().strip()
        self.use_bias = bool(use_bias)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        self._length = None
        self._in_ch = None
        self._out_len = None
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"Conv1DAsConv2D expects rank-3 input [B,L,C], got shape={input_shape}")
        length = input_shape[1]
        in_ch = input_shape[2]
        if length is None or in_ch is None:
            raise ValueError("Conv1DAsConv2D requires static L/C for TFLite/XNNPACK-friendly export.")
        if self.padding not in {"same", "valid"}:
            raise ValueError(f"Unsupported padding='{self.padding}'")
        if self.strides < 1:
            raise ValueError("strides must be >= 1")

        self._length = int(length)
        self._in_ch = int(in_ch)

        if self.padding == "same":
            self._out_len = int(np.ceil(self._length / self.strides))
        else:
            self._out_len = max(0, int(np.floor((self._length - self.kernel_size) / self.strides) + 1))

        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.kernel_size, self._in_ch, self.filters),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
            )
        super().build(input_shape)

    def call(self, inputs):
        if self._length is None or self._in_ch is None or self._out_len is None or self.kernel is None:
            raise RuntimeError("Conv1DAsConv2D is not built.")

        x = tf.reshape(inputs, [-1, 1, self._length, self._in_ch])
        k = tf.reshape(self.kernel, [1, self.kernel_size, self._in_ch, self.filters])
        y = tf.nn.conv2d(
            x,
            k,
            strides=[1, 1, self.strides, 1],
            padding=self.padding.upper(),
        )
        if self.use_bias and self.bias is not None:
            y = tf.nn.bias_add(y, self.bias)
        return tf.reshape(y, [-1, self._out_len, self.filters])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "use_bias": self.use_bias,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            }
        )
        return config


@register_keras_serializable(package="doccorner")
class GlobalAveragePool2DAsAvgPool(layers.Layer):
    """GlobalAveragePooling2D replacement with multiple implementations."""

    def __init__(self, impl: str = "mean", **kwargs):
        super().__init__(**kwargs)
        self.impl = str(impl).lower().strip()
        self._h = None
        self._w = None
        self._c = None
        self._filter_full = None
        self._filters_strided = None
        self._strides_strided = None
        self._filter_2x2 = None
        self._filter_2x1 = None
        self._filter_1x2 = None
        self._filter_final = None
        self._plan = None

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(f"GlobalAveragePool2DAsAvgPool expects rank-4 NHWC input, got shape={input_shape}")
        h, w, c = input_shape[1], input_shape[2], input_shape[3]
        if h is None or w is None or c is None:
            raise ValueError("GlobalAveragePool2DAsAvgPool requires static H/W/C for TFLite/XNNPACK-friendly export.")
        self._h = int(h)
        self._w = int(w)
        self._c = int(c)

        impl = self.impl
        if impl == "mean":
            super().build(input_shape)
            return

        if impl == "avgpool":
            super().build(input_shape)
            return

        if impl == "dwconv_full":
            k = np.ones((self._h, self._w, self._c, 1), dtype=np.float32) / float(self._h * self._w)
            self._filter_full = tf.constant(k, dtype=tf.float32, name=f"{self.name}_dwfilter_full")
            super().build(input_shape)
            return

        if impl == "dwconv_strided":
            hh = int(self._h)
            ww = int(self._w)
            filters = []
            strides = []
            factors = (8, 4, 2)
            while hh > 1 or ww > 1:
                if hh <= 1:
                    kh = 1
                else:
                    kh = None
                    for f in factors:
                        if hh % f == 0:
                            kh = f
                            break
                    if kh is None:
                        kh = int(hh)
                if ww <= 1:
                    kw = 1
                else:
                    kw = None
                    for f in factors:
                        if ww % f == 0:
                            kw = f
                            break
                if kw is None:
                    kw = int(ww)

                k = np.ones((kh, kw, self._c, 1), dtype=np.float32) / float(kh * kw)
                filters.append(tf.constant(k, dtype=tf.float32, name=f"{self.name}_dwfilter_{kh}x{kw}"))
                strides.append([1, kh, kw, 1])
                hh //= kh
                ww //= kw

            self._filters_strided = filters
            self._strides_strided = strides
            super().build(input_shape)
            return

        if impl == "dwconv_pyramid":
            self._filter_2x2 = tf.constant(
                np.ones((2, 2, self._c, 1), dtype=np.float32) * 0.25,
                dtype=tf.float32,
                name=f"{self.name}_dwfilter2x2",
            )
            self._filter_2x1 = tf.constant(
                np.ones((2, 1, self._c, 1), dtype=np.float32) * 0.5,
                dtype=tf.float32,
                name=f"{self.name}_dwfilter2x1",
            )
            self._filter_1x2 = tf.constant(
                np.ones((1, 2, self._c, 1), dtype=np.float32) * 0.5,
                dtype=tf.float32,
                name=f"{self.name}_dwfilter1x2",
            )

            plan = []
            hh = self._h
            ww = self._w
            while (hh > 1) or (ww > 1):
                if (hh > 1) and (ww > 1) and (hh % 2 == 0) and (ww % 2 == 0):
                    plan.append(("2x2", 2, 2))
                    hh //= 2
                    ww //= 2
                    continue
                if (hh > 1) and (hh % 2 == 0):
                    plan.append(("2x1", 2, 1))
                    hh //= 2
                    continue
                if (ww > 1) and (ww % 2 == 0):
                    plan.append(("1x2", 1, 2))
                    ww //= 2
                    continue
                break

            if hh > 1 or ww > 1:
                plan.append(("final", hh, ww))
                k = np.ones((hh, ww, self._c, 1), dtype=np.float32) / float(hh * ww)
                self._filter_final = tf.constant(k, dtype=tf.float32, name=f"{self.name}_dwfilter_final")
                hh = 1
                ww = 1

            if hh != 1 or ww != 1:
                raise RuntimeError(f"GlobalAveragePool2DAsAvgPool reduction plan did not reach 1x1: got {hh}x{ww}")

            self._plan = plan
            super().build(input_shape)
            return

        raise ValueError(f"Unsupported GlobalAveragePool2DAsAvgPool impl='{self.impl}'")
        super().build(input_shape)

    def call(self, inputs):
        if self._h is None or self._w is None or self._c is None:
            raise RuntimeError("GlobalAveragePool2DAsAvgPool is not built.")
        if self.impl == "mean":
            return tf.reduce_mean(inputs, axis=(1, 2))

        if self.impl == "avgpool":
            x = tf.nn.avg_pool2d(inputs, ksize=(self._h, self._w), strides=(1, 1), padding="VALID")
            return tf.reshape(x, [-1, self._c])

        if self.impl == "dwconv_full":
            if self._filter_full is None:
                raise RuntimeError("GlobalAveragePool2DAsAvgPool dwconv_full filter missing.")
            x = tf.nn.depthwise_conv2d(inputs, self._filter_full, strides=[1, 1, 1, 1], padding="VALID")
            return tf.reshape(x, [-1, self._c])

        if self.impl == "dwconv_strided":
            if self._filters_strided is None or self._strides_strided is None:
                raise RuntimeError("GlobalAveragePool2DAsAvgPool dwconv_strided filters missing.")
            x = inputs
            for f, s in zip(self._filters_strided, self._strides_strided, strict=False):
                x = tf.nn.depthwise_conv2d(x, f, strides=s, padding="VALID")
            return tf.reshape(x, [-1, self._c])

        if self.impl == "dwconv_pyramid":
            if self._plan is None:
                raise RuntimeError("GlobalAveragePool2DAsAvgPool dwconv_pyramid plan missing.")
            if self._filter_2x2 is None or self._filter_2x1 is None or self._filter_1x2 is None:
                raise RuntimeError("GlobalAveragePool2DAsAvgPool dwconv_pyramid filters missing.")

            x = inputs
            for kind, _kh, _kw in self._plan:
                if kind == "2x2":
                    x = tf.nn.depthwise_conv2d(x, self._filter_2x2, strides=[1, 2, 2, 1], padding="VALID")
                elif kind == "2x1":
                    x = tf.nn.depthwise_conv2d(x, self._filter_2x1, strides=[1, 2, 1, 1], padding="VALID")
                elif kind == "1x2":
                    x = tf.nn.depthwise_conv2d(x, self._filter_1x2, strides=[1, 1, 2, 1], padding="VALID")
                elif kind == "final":
                    if self._filter_final is None:
                        raise RuntimeError("GlobalAveragePool2DAsAvgPool dwconv_pyramid final filter missing.")
                    x = tf.nn.depthwise_conv2d(x, self._filter_final, strides=[1, 1, 1, 1], padding="VALID")
                else:
                    raise RuntimeError(f"Unexpected reduction kind='{kind}'")
            return tf.reshape(x, [-1, self._c])

        raise ValueError(f"Unsupported GlobalAveragePool2DAsAvgPool impl='{self.impl}'")

    def get_config(self):
        config = super().get_config()
        config.update({"impl": self.impl})
        return config


@register_keras_serializable(package="doccorner")
class NearestUpsample2x(layers.Layer):
    """2x nearest-neighbor upsampling using only RESHAPE+MUL broadcasting (XNNPACK-friendly)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._h = None
        self._w = None
        self._c = None
        self._ones_w = None
        self._ones_h = None

    def build(self, input_shape):
        h, w, c = input_shape[1], input_shape[2], input_shape[3]
        if h is None or w is None or c is None:
            raise ValueError("NearestUpsample2x requires static H/W/C for TFLite/XNNPACK-friendly export.")
        self._h = int(h)
        self._w = int(w)
        self._c = int(c)

        self._ones_w = tf.constant(
            np.ones((1, 1, 1, 2, 1), dtype=np.float32),
            dtype=tf.float32,
            name=f"{self.name}_ones_w",
        )
        self._ones_h = tf.constant(
            np.ones((1, 1, 2, 1, 1), dtype=np.float32),
            dtype=tf.float32,
            name=f"{self.name}_ones_h",
        )
        super().build(input_shape)

    def call(self, inputs):
        if self._h is None or self._w is None or self._c is None or self._ones_w is None or self._ones_h is None:
            raise RuntimeError("NearestUpsample2x is not built.")

        # Repeat along width: [B,H,W,C] -> [B,H,W,2,C] -> [B,H,2W,C]
        x = tf.reshape(inputs, [-1, self._h, self._w, 1, self._c])
        x = x * self._ones_w
        x = tf.reshape(x, [-1, self._h, self._w * 2, self._c])

        # Repeat along height: [B,H,2W,C] -> [B,H,2,2W,C] -> [B,2H,2W,C]
        x = tf.reshape(x, [-1, self._h, 1, self._w * 2, self._c])
        x = x * self._ones_h
        return tf.reshape(x, [-1, self._h * 2, self._w * 2, self._c])

    def get_config(self):
        return super().get_config()


@register_keras_serializable(package="doccorner")
class SimCCDecode(layers.Layer):
    """Decode SimCC logits to normalized coordinates using soft-argmax."""

    def __init__(self, num_bins: int, tau: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.num_bins = int(num_bins)
        self.tau = float(tau)
        self._centers_col = None

    def build(self, input_shape):
        # Constant bin centers in [0,1], used for expectation via matmul (avoids SUM/TILE).
        centers = np.linspace(0.0, 1.0, self.num_bins, dtype=np.float32).reshape(self.num_bins, 1)
        self._centers_col = tf.constant(centers, dtype=tf.float32, name=f"{self.name}_centers")
        super().build(input_shape)

    def call(self, inputs):
        if self._centers_col is None:
            raise RuntimeError("SimCCDecode is not built.")

        sx, sy = inputs  # [B, 4, num_bins]

        sx = tf.cast(sx, tf.float32)
        sy = tf.cast(sy, tf.float32)

        px = tf.nn.softmax(sx / self.tau, axis=-1)
        py = tf.nn.softmax(sy / self.tau, axis=-1)

        # Compute expected value with a constant matmul:
        #   [B*4, num_bins] @ [num_bins, 1] -> [B*4, 1] -> [B, 4]
        px2 = tf.reshape(px, [-1, self.num_bins])
        py2 = tf.reshape(py, [-1, self.num_bins])
        x = tf.reshape(tf.matmul(px2, self._centers_col), [-1, 4])
        y = tf.reshape(tf.matmul(py2, self._centers_col), [-1, 4])

        # Interleave without STRIDED_SLICE / PACK:
        # [B,4] -> [B,4,1] concat -> [B,4,2] reshape -> [B,8] (x0,y0,x1,y1,...)
        xy = tf.concat([tf.reshape(x, [-1, 4, 1]), tf.reshape(y, [-1, 4, 1])], axis=-1)
        coords = tf.reshape(xy, [-1, 8])
        return tf.clip_by_value(coords, 0.0, 1.0)

    def get_config(self):
        config = super().get_config()
        config.update({"num_bins": self.num_bins, "tau": self.tau})
        return config


def _get_feature_layers(backbone, img_size: int):
    """
    Extract multi-scale feature layers from a conv backbone.

    Returns outputs at different scales:
    - C2: img/4, C3: img/8, C4: img/16, C5: img/32
    """
    c2 = c3 = c4 = c5 = None
    c2_hw = img_size // 4
    c3_hw = img_size // 8
    c4_hw = img_size // 16
    c5_hw = img_size // 32

    for layer in backbone.layers:
        out = layer.output
        if not hasattr(out, 'shape') or len(out.shape) != 4:
            continue

        _, h, w, c = out.shape
        if h == 1 and w == 1:
            continue
        if h is None or w is None:
            continue
        if h != w:
            continue

        if h == c2_hw and w == c2_hw:
            c2 = out
        elif h == c3_hw and w == c3_hw:
            c3 = out
        elif h == c4_hw and w == c4_hw:
            c4 = out
        elif h == c5_hw and w == c5_hw:
            c5 = out

    if c2 is None or c3 is None or c4 is None or c5 is None:
        raise ValueError("Could not find all feature scales in backbone")

    return c2, c3, c4, c5


def _build_backbone(
    inp,
    backbone: str,
    alpha: float,
    backbone_weights,
    backbone_minimalistic: bool,
    backbone_include_preprocessing: bool,
) -> keras.Model:
    backbone_key = backbone.strip().lower().replace("-", "_")

    if backbone_key in {"mobilenetv3_small", "mobilenet_v3_small", "mnetv3_small"}:
        return keras.applications.MobileNetV3Small(
            input_tensor=inp,
            include_top=False,
            weights=backbone_weights,
            alpha=alpha,
            minimalistic=backbone_minimalistic,
            include_preprocessing=backbone_include_preprocessing,
        )

    if backbone_key in {"mobilenetv3_large", "mobilenet_v3_large", "mnetv3_large"}:
        return keras.applications.MobileNetV3Large(
            input_tensor=inp,
            include_top=False,
            weights=backbone_weights,
            alpha=alpha,
            minimalistic=backbone_minimalistic,
            include_preprocessing=backbone_include_preprocessing,
        )

    if backbone_key in {"mobilenetv2", "mobilenet_v2", "mnetv2"}:
        if backbone_minimalistic:
            raise ValueError("backbone_minimalistic is only supported for MobileNetV3.")
        if backbone_include_preprocessing:
            x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="backbone_preprocess")(inp)
            input_tensor = x
        else:
            input_tensor = inp
        return keras.applications.MobileNetV2(
            input_tensor=input_tensor,
            include_top=False,
            weights=backbone_weights,
            alpha=alpha,
        )

    raise ValueError(
        f"Unsupported backbone='{backbone}'. "
        "Use one of: mobilenetv2, mobilenetv3_small, mobilenetv3_large."
    )


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
    backbone: str = "mobilenetv3_small",
    backbone_minimalistic: bool = False,
    backbone_include_preprocessing: bool = False,
    conv1d_as_conv2d: bool = False,
    axis_mean_impl: str = "mean",
    global_pool_impl: str = "mean",
    score_pool_impl: str | None = None,
    simcc_output_layout: str = "corners_first",
):
    """
    Build DocCornerNetV3 model with correct SimCC (MMPose style).

    SimCC predicts X and Y coordinates directly as 1D classification:
    - Features → FC → simcc_x [B, 4, num_bins]
    - Features → FC → simcc_y [B, 4, num_bins]
    - Soft-argmax → coords [B, 8]

    Args:
        backbone: Backbone architecture ('mobilenetv3_small' or 'mobilenetv3_large')
        alpha: Backbone width multiplier (alpha)
        backbone_minimalistic: Use MobileNetV3 minimalistic variant
        backbone_include_preprocessing: Use built-in backbone preprocessing layers
        backbone_weights: Backbone initialization weights ('imagenet' or None)
        fpn_ch: FPN intermediate channels
        simcc_ch: SimCC head hidden channels
        img_size: Input image size (default 224)
        num_bins: Number of bins for coordinate classification
        tau: Temperature for softmax in coordinate decode
        score_init_bias: Initial bias for score head
        axis_mean_impl: Axis reduction impl ('mean', 'avgpool', 'dwconv_full', 'dwconv_strided', 'dwconv_pyramid')
        global_pool_impl: Global pooling impl ('mean', 'avgpool', 'dwconv_full', 'dwconv_strided', 'dwconv_pyramid')
        score_pool_impl: Pooling impl override for score_gap (defaults to global_pool_impl)
        simcc_output_layout: SimCC tensor layout ('corners_first' => [B,4,num_bins], 'bins_first' => [B,num_bins,4])

    Returns:
        Keras Model with outputs dict
    """
    inp = keras.Input((img_size, img_size, 3), name="image")
    conv1d_layer = Conv1DAsConv2D if bool(conv1d_as_conv2d) else layers.Conv1D
    score_pool_impl_resolved = global_pool_impl if score_pool_impl in (None, "") else str(score_pool_impl)
    simcc_layout = str(simcc_output_layout).lower().strip()

    # =========================================================================
    # Backbone (optionally ImageNet pretrained)
    # =========================================================================
    backbone_model = _build_backbone(
        inp=inp,
        backbone=backbone,
        alpha=alpha,
        backbone_weights=backbone_weights,
        backbone_minimalistic=backbone_minimalistic,
        backbone_include_preprocessing=backbone_include_preprocessing,
    )

    c2, c3, c4, c5 = _get_feature_layers(backbone_model, img_size=img_size)

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
    # 2x nearest without RESIZE_NEAREST_NEIGHBOR / TILE, so XNNPACK can fully delegate it.
    p4_up = NearestUpsample2x(name="fpn_p4_up")(p4)
    p3 = layers.Add(name="fpn_p3_add")([p3, p4_up])
    p3 = _separable_conv_block(p3, fpn_ch, "fpn_p3_refine")

    p3_up = NearestUpsample2x(name="fpn_p3_up")(p3)
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
    x_marginal = AxisMean(axis=1, impl=axis_mean_impl, name="x_marginal_pool")(p_fused)  # [B, 56, fpn_ch]

    # Generate Y marginals: reduce along X axis (horizontal pooling)
    # [B, 56, 56, fpn_ch] -> [B, 56, fpn_ch] via mean along axis 2
    y_marginal = AxisMean(axis=2, impl=axis_mean_impl, name="y_marginal_pool")(p_fused)  # [B, 56, fpn_ch]

    # Upsample marginals to num_bins resolution
    # [B, 56, fpn_ch] -> [B, num_bins, fpn_ch]
    x_marginal = Resize1D(target_length=num_bins, name="x_marginal_resize")(x_marginal)  # [B, 224, fpn_ch]
    y_marginal = Resize1D(target_length=num_bins, name="y_marginal_resize")(y_marginal)  # [B, 224, fpn_ch]

    # 1D convolutions for coordinate refinement along each axis
    x_feat = conv1d_layer(simcc_ch, 5, padding="same", name="simcc_x_conv1")(x_marginal)
    x_feat = layers.BatchNormalization(name="simcc_x_bn1")(x_feat)
    x_feat = layers.ReLU(name="simcc_x_relu1")(x_feat)
    x_feat = conv1d_layer(simcc_ch // 2, 3, padding="same", name="simcc_x_conv2")(x_feat)
    x_feat = layers.BatchNormalization(name="simcc_x_bn2")(x_feat)
    x_feat = layers.ReLU(name="simcc_x_relu2")(x_feat)

    y_feat = conv1d_layer(simcc_ch, 5, padding="same", name="simcc_y_conv1")(y_marginal)
    y_feat = layers.BatchNormalization(name="simcc_y_bn1")(y_feat)
    y_feat = layers.ReLU(name="simcc_y_relu1")(y_feat)
    y_feat = conv1d_layer(simcc_ch // 2, 3, padding="same", name="simcc_y_conv2")(y_feat)
    y_feat = layers.BatchNormalization(name="simcc_y_bn2")(y_feat)
    y_feat = layers.ReLU(name="simcc_y_relu2")(y_feat)

    # Global context via GAP (auxiliary, for corner disambiguation)
    global_feat = GlobalAveragePool2DAsAvgPool(impl=global_pool_impl, name="simcc_global_gap")(p_fused)  # [B, fpn_ch]
    global_feat = layers.Dense(simcc_ch // 2, name="simcc_global_fc")(global_feat)
    global_feat = layers.ReLU(name="simcc_global_relu")(global_feat)

    # Broadcast global features to each position
    # [B, simcc_ch//2] -> [B, num_bins, simcc_ch//2]
    global_x = Broadcast1D(target_length=num_bins, name="global_x_broadcast")(global_feat)
    global_y = Broadcast1D(target_length=num_bins, name="global_y_broadcast")(global_feat)

    # Concatenate local and global features
    x_feat = layers.Concatenate(name="simcc_x_cat")([x_feat, global_x])  # [B, 224, simcc_ch]
    y_feat = layers.Concatenate(name="simcc_y_cat")([y_feat, global_y])  # [B, 224, simcc_ch]

    # Output heads: 4 corners for each axis
    # [B, num_bins, simcc_ch] -> [B, num_bins, 4]
    simcc_x = conv1d_layer(
        4, 1,
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=keras.initializers.Zeros(),
        name="simcc_x_out"
    )(x_feat)  # [B, 224, 4]
    if simcc_layout in {"corners_first", "corner_first", "channels_first", "first"}:
        simcc_x = layers.Permute((2, 1), name="simcc_x")(simcc_x)  # [B, 4, num_bins]
    elif simcc_layout in {"bins_first", "bin_first", "channels_last", "last"}:
        # Keep as [B, num_bins, 4] (no TRANSPOSE op).
        pass
    else:
        raise ValueError(f"Unsupported simcc_output_layout='{simcc_output_layout}'")

    simcc_y = conv1d_layer(
        4, 1,
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=keras.initializers.Zeros(),
        name="simcc_y_out"
    )(y_feat)  # [B, 224, 4]
    if simcc_layout in {"corners_first", "corner_first", "channels_first", "first"}:
        simcc_y = layers.Permute((2, 1), name="simcc_y")(simcc_y)  # [B, 4, num_bins]
    elif simcc_layout in {"bins_first", "bin_first", "channels_last", "last"}:
        # Keep as [B, num_bins, 4] (no TRANSPOSE op).
        pass
    else:
        raise ValueError(f"Unsupported simcc_output_layout='{simcc_output_layout}'")

    coords = SimCCDecode(num_bins=num_bins, tau=tau, name="coords")([simcc_x, simcc_y])

    # =========================================================================
    # Score Head: Document presence classification
    # =========================================================================
    score_features = GlobalAveragePool2DAsAvgPool(impl=score_pool_impl_resolved, name="score_gap")(c5)
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
    backbone: str = "mobilenetv3_small",
    backbone_minimalistic: bool = False,
    backbone_include_preprocessing: bool = False,
    conv1d_as_conv2d: bool = False,
    axis_mean_impl: str = "mean",
    global_pool_impl: str = "mean",
    score_pool_impl: str | None = None,
    simcc_output_layout: str = "corners_first",
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
        backbone=backbone,
        backbone_minimalistic=backbone_minimalistic,
        backbone_include_preprocessing=backbone_include_preprocessing,
        conv1d_as_conv2d=conv1d_as_conv2d,
        axis_mean_impl=axis_mean_impl,
        global_pool_impl=global_pool_impl,
        score_pool_impl=score_pool_impl,
        simcc_output_layout=simcc_output_layout,
    )


def build_doccorner_simcc_v3_inference(**model_kwargs) -> keras.Model:
    """
    Backwards-compatible inference-model builder.

    Returns a model with outputs: [coords, score_logit].
    """
    train_model = create_model(**model_kwargs)
    return create_inference_model(train_model)


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
