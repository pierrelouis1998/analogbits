from abc import ABC

import keras
import tensorflow as tf


def get_model(input_shape: tf.TensorShape = None, model_name='dummy', trainable=True):
    """Return tensorflow model"""
    model = None

    if model_name == 'dummy':
        model = dummy_model(input_shape)

    elif model_name == 'unet':
        model = TransUNet(input_shape=input_shape)

    model.trainable = trainable

    return model


class TransUNet(tf.keras.layers.Layer):
    """TransUNet architecture."""

    def __init__(self,
                 n_res_blocks=(4, 4, 4),
                 kernel_sizes=(3, 3, 3),
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(n_res_blocks)

    def call(self, x, t, cembs, training, return_logits=False):
        """x in (bsz, h, w, c), t in (bsz,) or (bsz, k), cembs in (bsz, s, d)."""
        assert x.shape.rank == 4
        last = x.shape[-1] // 2
        return x[:, :, :, :last]


def dummy_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=input_shape),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(input_shape[-1])
    ])
    return model
