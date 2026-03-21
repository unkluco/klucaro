"""
model_fcn.py — Fully Convolutional Network cho Caro

Thay đổi so với model gốc:
  1. Policy head: Conv2D(1,1×1) thay Dense(bs²) → linh hoạt board size
  2. Input: 3 channels (quân ta, quân địch, ô trống) thay 1 channel
"""

import tensorflow as tf
from tensorflow.keras import layers as L
import numpy as np
import keras as _keras

from config import WIN_LEN


# ================================================================
# ResBlock
# ================================================================

@_keras.saving.register_keras_serializable(package="CaroAZ")
class ResBlock(L.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv1 = L.Conv2D(filters, 3, padding="same", use_bias=False)
        self.bn1   = L.BatchNormalization()
        self.relu1 = L.ReLU()
        self.conv2 = L.Conv2D(filters, 3, padding="same", use_bias=False)
        self.bn2   = L.BatchNormalization()
        self.add   = L.Add()
        self.relu2 = L.ReLU()

    def build(self, input_shape):
        self.conv1.build(input_shape)
        after_conv = (input_shape[0], input_shape[1], input_shape[2], self.filters)
        self.bn1.build(after_conv)
        self.conv2.build(after_conv)
        self.bn2.build(after_conv)
        super().build(input_shape)

    def call(self, x, training=None):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.add([out, residual])
        return self.relu2(out)

    def get_config(self):
        cfg = super().get_config()
        cfg["filters"] = self.filters
        return cfg


# ================================================================
# BUILD MODEL
# ================================================================
#
#  Input (bs, bs, 3)  ← 3 planes: quân ta, quân địch, ô trống
#      ↓ Stem: Conv2D(F,3×3) → BN → ReLU
#      ↓ Trunk: n_res × ResBlock(F)
#      ├── Policy head: Conv2D(2) → BN → ReLU → Conv2D(1,1×1) → Flatten → Softmax
#      └── Value head:  Conv2D(1) → BN → ReLU → GlobalAvgPool → Dense(64) → Dense(1,tanh)

def build_model(board_size=None, base_filters=128, n_res=10):
    inp = L.Input((board_size, board_size, 3))

    # Stem
    x = L.Conv2D(base_filters, 3, padding="same", use_bias=False)(inp)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)

    # Trunk
    for _ in range(n_res):
        x = ResBlock(base_filters)(x)

    # Policy head (fully convolutional)
    policy = L.Conv2D(2, 1, padding="same", use_bias=False)(x)
    policy = L.BatchNormalization()(policy)
    policy = L.ReLU()(policy)
    policy = L.Conv2D(1, 1, padding="same", use_bias=True)(policy)
    policy = L.Reshape((-1,))(policy)
    policy = L.Softmax(name="policy_softmax")(policy)

    # Value head
    value = L.Conv2D(1, 1, padding="same", use_bias=False)(x)
    value = L.BatchNormalization()(value)
    value = L.ReLU()(value)
    value = L.GlobalAveragePooling2D()(value)
    value = L.Dense(64, activation="relu")(value)
    value = L.Dense(1,  activation="tanh")(value)

    return tf.keras.Model(inputs=inp, outputs=[policy, value], name="CaroNet_FCN")


# ================================================================
# HELPERS
# ================================================================

def reshape_policy(policy_flat, board_size):
    """Reshape policy từ flat (B, H*W) về (B, H, W)."""
    if isinstance(policy_flat, np.ndarray):
        return policy_flat.reshape(-1, board_size, board_size)
    return tf.reshape(policy_flat, (-1, board_size, board_size))


def transfer_weights(src_model, dst_model):
    """Copy weights giữa 2 model cùng kiến trúc, khác board_size."""
    src_w = src_model.get_weights()
    dst_w = dst_model.get_weights()
    assert len(src_w) == len(dst_w), \
        f"Số weight tensors không khớp: {len(src_w)} vs {len(dst_w)}"
    for i, (sw, dw) in enumerate(zip(src_w, dst_w)):
        assert sw.shape == dw.shape, \
            f"Weight #{i} shape không khớp: {sw.shape} vs {dw.shape}"
    dst_model.set_weights(src_w)
    print(f"✓ Transferred {len(src_w)} weight tensors")
