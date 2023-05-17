from abc import ABC

import keras
import tensorflow as tf
from dotwiz import DotWiz

from models.layers_spec import ScalarEmbedding, get_variable_initializer, get_norm, ConvUnit, Transformer
from utils.bit_encoding import int2bits


def get_model(input_shape: tf.TensorShape = None, model_name='dummy', trainable=True):
    """Return tensorflow model"""
    model = None

    if model_name == 'dummy':
        model = dummy_model(input_shape)

    elif model_name == 'unet':
        model = TransUNet(
            transformer_blocks=0
        )
    elif model_name == 'custom':
        model = MyModel()

    model.trainable = trainable

    return model


class TransUNet(tf.keras.layers.Layer):
    """TransUNet architecture."""

    def __init__(self,
                 out_dim=3,
                 dim=256,
                 in_strides=1,
                 in_kernel_size=3,
                 out_kernel_size=3,
                 n_res_blocks=(4, 4, 4),
                 ch_multipliers=(1, 1, 1),
                 kernel_sizes=(3, 3, 3),
                 n_mlp_blocks=0,
                 dropout=0.2,
                 mhsa_resolutions=(16, 8),
                 per_head_dim=64,
                 transformer_dim=256,
                 transformer_strides=2,
                 transformer_blocks=4,
                 pos_encoding='learned',
                 conditioning=False,
                 time_scaling=1e4,
                 norm_type='group_norm',
                 outp_softmax_groups=0,
                 b_scale=1.,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(n_res_blocks)
        initial_dictionary = {
            'out_dim': out_dim,
            'dim': dim,
            'in_strides': in_strides,
            'in_kernel_size': in_kernel_size,
            'out_kernel_size': out_kernel_size,
            'n_res_blocks': n_res_blocks,
            'kernel_sizes': kernel_sizes,
            'ch_multipliers': ch_multipliers,
            'n_mlp_blocks': n_mlp_blocks,
            'dropout': dropout,
            'mhsa_resolutions': mhsa_resolutions,
            'per_head_dim': per_head_dim,
            'transformer_dim': transformer_dim,
            'transformer_strides': transformer_strides,
            'transformer_blocks': transformer_blocks,
            'pos_encoding': pos_encoding,
            'conditioning': conditioning,
            'time_scaling': time_scaling,
            'norm_type': norm_type,
            'outp_softmax_groups': outp_softmax_groups,
            'b_scale': b_scale,
        }
        self.config = DotWiz(initial_dictionary)

    def build(self, input_shape):
        config = self.config
        self.in_dim = input_shape[-1]
        self.time_emb = ScalarEmbedding(
            dim=config.dim,
            scaling=config.time_scaling,
            name='time_emb')

        self.conv_in = tf.keras.layers.Conv2D(
            filters=config.dim * config.ch_multipliers[0],
            kernel_size=[config.in_kernel_size, config.in_kernel_size],
            strides=[config.in_strides, config.in_strides],
            padding='SAME',
            use_bias=True,
            kernel_initializer=get_variable_initializer(1.0),
            name='conv_in')
        self.group_norm_in = get_norm(
            'group_norm',
            num_groups=min(config.dim * config.ch_multipliers[0] // 4, 32),
            name='group_norm_in')

        # dblocks and ublocks
        self.up_units = {}
        self.dn_units = {}
        for i, (n_res_blocks, kernel_size, ch_multiplier) in enumerate(zip(
                config.n_res_blocks,
                config.kernel_sizes,
                config.ch_multipliers)):
            self.dn_units[str(i)] = ConvUnit(
                config,
                n_res_blocks=n_res_blocks,
                kernel_size=kernel_size,
                ch_multiplier=ch_multiplier,
                expect_lateral=False,
                up_dn_sample='dn' if i != len(config.n_res_blocks) - 1 else 'none',
                name=f'dn_unit_{i}')
            self.up_units[str(i)] = ConvUnit(
                config,
                n_res_blocks=n_res_blocks + 1,
                kernel_size=kernel_size,
                ch_multiplier=ch_multiplier,
                expect_lateral=True,
                up_dn_sample='up' if i != 0 else 'none',
                name=f'up_unit_{i}')

        self.up_mlp_units = {}
        self.dn_mlp_units = {}
        for i in range(config.n_mlp_blocks):
            self.dn_mlp_units[str(i)] = ConvUnit(
                config,
                n_res_blocks=config.n_mlp_blocks,
                kernel_size=1,
                ch_multiplier=1,
                expect_lateral=False,
                up_dn_sample='none',
                name=f'dn_mlp_unit_{i}')
            self.up_mlp_units[str(i)] = ConvUnit(
                config,
                n_res_blocks=config.n_mlp_blocks,
                kernel_size=1,
                ch_multiplier=1,
                expect_lateral=True,
                up_dn_sample='none',
                name=f'up_mlp_unit_{i}')

        if config.transformer_blocks > 0:
            self.transformer = Transformer(
                dim=config.transformer_dim,
                strides=config.transformer_strides,
                num_layers=config.transformer_blocks,
                num_heads=config.transformer_dim // config.per_head_dim,
                mlp_ratio=4,
                drop_units=config.dropout,
                drop_att=0.,
                drop_path=0.,
                pos_encoding=config.pos_encoding,
                conditioning=config.conditioning,
                name='transformer')

        self.group_norm_out = get_norm(
            'group_norm',
            num_groups=min(config.dim // 4, 32),
            name='group_norm_out')
        if config.outp_softmax_groups == 0:
            self.conv_out = tf.keras.layers.Conv2D(
                filters=config.out_dim * config.in_strides ** 2,
                kernel_size=[config.out_kernel_size, config.out_kernel_size],
                strides=[1, 1],
                padding='SAME',
                use_bias=True,
                kernel_initializer=get_variable_initializer(),
                name='conv_out')
        else:
            nbits = config.out_dim // config.outp_softmax_groups
            self.conv_out = tf.keras.layers.Conv2D(
                filters=2 ** nbits * config.outp_softmax_groups,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding='SAME',
                use_bias=True,
                kernel_initializer=get_variable_initializer(),
                name='conv_out')
            self.bits = int2bits(tf.range(2 ** nbits), nbits, tf.float32)

        self.linear = tf.keras.layers.Dense(24)

    def call(self, x, t, cembs, training, return_logits=False):
        """x in (bsz, h, w, c), t in (bsz,) or (bsz, k), cembs in (bsz, s, d)."""
        config = self.config
        emb = self.time_emb(t)
        x = self.group_norm_in(self.conv_in(x))

        x_list = [x]

        for i in range(config.n_mlp_blocks):
            x = self.dn_mlp_units[str(i)](x, emb, cembs, x_list, training)

        for i in range(len(config.n_res_blocks)):
            x = self.dn_units[str(i)](x, emb, cembs, x_list, training)

        if config.transformer_blocks > 0:
            x = self.transformer(x, emb, cembs, training)
        # for x_ in x_list + [x]:
        #   print('x shape', x_.shape)
        for i in range(len(config.n_res_blocks))[::-1]:
            x = self.up_units[str(i)](x, emb, cembs, x_list, training)

        for i in range(config.n_mlp_blocks)[::-1]:
            x = self.up_mlp_units[str(i)](x, emb, cembs, x_list, training)

        assert not x_list, 'x_list should be fully utilized'

        x = self.conv_out(tf.nn.silu(self.group_norm_out(x)))
        if config.in_strides > 1:
            x = tf.nn.depth_to_space(x, config.in_strides)

        x = self.linear(x)

        if config.outp_softmax_groups > 0:
            xs_ori = xs = tf.split(x, config.outp_softmax_groups, -1)
            xs = [tf.einsum('bhwd,do->bhwo', tf.nn.softmax(x), self.bits) for x in xs]
            x = (tf.concat(xs, -1) * 2 - 1) * config.b_scale
            if return_logits:
                x = tf.stack(xs_ori, -2)  # (bsz, h, w, groups, n_class_per_group)
        # print("x out :", x.shape)
        return x


class MyModel(tf.keras.layers.Layer):
    def __init__(
            self,
            out_dim=3,
            dim=256,
            time_scaling=1e4,
            b_scale=1.,
    ):
        super(MyModel, self).__init__()
        self.emb_proj = None
        self.linear_bis = None
        self.conv1 = None
        self.linear = None
        self.time_emb = None
        self.in_dim = None
        initial_dictionary = {
            'dim': dim,
            'out_dim': out_dim,
            'time_scaling': time_scaling,
            'b_scale': b_scale,
        }
        self.config = DotWiz(initial_dictionary)

    def build(self, input_shape):
        config = self.config
        self.in_dim = input_shape[-1]
        self.time_emb = ScalarEmbedding(
            dim=self.in_dim,
            scaling=config.time_scaling,
            name='time_emb',
            expansion=1
        )
        self.linear = tf.keras.layers.Dense(24)
        self.conv1 = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same'),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same'),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(24, 3, activation='relu', padding='same'),
            tf.keras.layers.Dense(24),
        ])
        self.emb_proj = tf.keras.layers.Dense(
            units=self.in_dim * 2,
            kernel_initializer=get_variable_initializer(1.0),
            name='emb_proj')

    def call(self, x, t, cembs, training, return_logits=False):
        config = self.config
        emb = self.time_emb(t)
        emb = self.emb_proj(emb)[:, None, None, :]
        emb_scale, emb_shift = tf.split(emb, 2, axis=-1)
        emb_scale += 1.0

        x = x * emb_scale + emb_shift
        x = self.conv1(x, training=training)
        x = self.linear(x, training=training)
        return x


def dummy_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=input_shape),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(input_shape[-1])
    ])
    return model
