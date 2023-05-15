# coding=utf-8
# Copyright 2022 The Pix2Seq Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The discrete/categorical image diffusion model."""
import sys
sys.path.append('.')
import utils
from utils import diffusion_utils
from models.dummy_model import get_model
import tensorflow as tf

from utils.bit_encoding import unfold_rgb, rgb2bit, bit2rgb, fold_rgb, get_x_channels


class Model(tf.keras.models.Model):
    """A model."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        image_size = config.dataset.image_size
        self.image_size = image_size
        self.num_classes = config.dataset.num_classes
        config = config.model
        self.config = config
        self.scheduler = diffusion_utils.Scheduler(config.train_schedule)
        if config.x0_clip == 'auto':
            self.x0_clip = '{},{}'.format(-config.b_scale, config.b_scale)
        else:
            self.x0_clip = config.x0_clip
        m_kwargs = config.m_kwargs
        self.x_channels = get_x_channels(config.b_type)
        self.denoiser = get_model(**m_kwargs)
        self.denoiser_ema = get_model(**m_kwargs, trainable=False)
        # Obtain hidden shapes for latent self conditioning.
        self.hidden_shapes = getattr(self.denoiser, 'hidden_shapes', None)
        if self.hidden_shapes is not None:  # latent self-cond
            assert config.self_cond not in ['x', 'eps', 'auto']

    def get_cond_denoise(self, labels, for_loss=False):
        config = self.config

        def cond_denoise(x, gamma, training, drop_label=False):
            gamma = tf.reshape(gamma, [-1])
            cond = None
            if config.conditional == 'class':
                cond_dropout = config.get('cond_dropout', 0.)
                labels_w = 1.
                if training and cond_dropout > 0:
                    labels_w = tf.random.uniform([tf.shape(labels)[0], 1]) > cond_dropout
                    labels_w = tf.cast(labels_w, tf.float32)
                if drop_label:
                    labels_w = 0.
                if config.arch_name == 'transunet':  # Merge one-hot label with gamma.
                    gamma = tf.concat([gamma[..., tf.newaxis], labels * labels_w], -1)
                else:
                    cond = labels * labels_w
            elif config.conditional != 'none':
                raise ValueError(f'Unknown conditional {config.conditional}')
            return self.denoise(x, gamma, cond, training, for_loss=for_loss)

        return cond_denoise

    def denoise(self, x, gamma, cond, training, for_loss=False):
        """gamma should be (bsz, ) or (bsz, d)."""
        assert gamma.shape.rank <= 2
        config = self.config
        if not hasattr(self, 'denoise_x_shape'):
            if isinstance(x, tuple) or isinstance(x, list):
                self.denoise_x_shape = tuple(tf.shape(x_) for x_ in x)
            else:
                self.denoise_x_shape = tf.shape(x)
            self.denoise_gamma_shape = tf.shape(gamma)
            self.cond_shape = None if cond is None else tf.shape(cond)
        denoiser = self.denoiser if training else self.denoiser_ema
        if self.config.normalize_noisy_input:
            if isinstance(x, tuple) or isinstance(x, list):
                x = list(x)
                x[0] /= tf.math.reduce_std(
                    x[0], list(range(1, x[0].shape.ndims)), keepdims=True)
            else:
                x /= tf.math.reduce_std(x, list(range(1, x.shape.ndims)), keepdims=True)
        x = denoiser(x, gamma, cond, training=training)
        if config.pred_type == 'x_sigmoid_xent':
            x = x if for_loss else (tf.nn.sigmoid(x) * 2 - 1) * config.b_scale
        if config.pred_type == 'x_softmax_xent':
            if isinstance(x, tuple) or isinstance(x, list):
                x = list(x)
                x[0] = unfold_rgb(x[0])
                x[0] = x[0] if for_loss else (
                                                 fold_rgb(tf.nn.softmax(x[0]) * 2 - 1)) * config.b_scale
                x = tuple(x)
            else:
                x = unfold_rgb(x)
                x = x if for_loss else (
                                               fold_rgb(tf.nn.softmax(x)) * 2 - 1) * config.b_scale
        return x

    def sample(self, num_samples=100, iterations=100, method='ddim', **kwargs):
        config = self.config
        samples_shape = [
            num_samples, self.image_size[0], self.image_size[1], self.x_channels]
        if config.conditional == 'class':
            labels = tf.random.uniform(
                [num_samples], 0, self.num_classes, dtype=tf.int32)
            labels = tf.one_hot(labels, self.num_classes)
        else:
            labels = None
        samples = self.scheduler.generate(
            self.get_cond_denoise(labels),
            iterations,
            samples_shape,
            hidden_shapes=self.hidden_shapes,
            pred_type=config.pred_type,
            schedule=config.infer_schedule,
            td=config.td,
            x0_clip=self.x0_clip,
            self_cond=config.self_cond,
            guidance=config.guidance,
            sampler_name=method)
        return bit2rgb(samples, config.b_type)

    def noise_denoise(self, images, labels, time_step=None, training=True):
        config = self.config
        images = rgb2bit(images, config.b_type, config.b_scale, self.x_channels)
        images_noised, noise, _, gamma = self.scheduler.add_noise(
            images, time_step=time_step)
        if config.self_cond != 'none':
            sc_rate = config.get('self_cond_rate', 0.5)
            self_cond_by_masking = config.get('self_cond_by_masking', False)
            if self_cond_by_masking:
                sc_drop_rate = 1. - sc_rate
                num_sc_examples = tf.shape(images)[0]
            else:
                sc_drop_rate = 0.
                num_sc_examples = tf.cast(
                    tf.cast(tf.shape(images)[0], tf.float32) * sc_rate, tf.int32)
            cond_denoise = self.get_cond_denoise(labels[:num_sc_examples])
            if self.hidden_shapes is None:  # data self-cond, return is a tensor.
                denoise_inputs = diffusion_utils.add_self_cond_estimate(
                    images_noised, gamma, cond_denoise, config.pred_type,
                    config.self_cond, self.x0_clip, num_sc_examples,
                    drop_rate=sc_drop_rate, training=training)
            else:  # latent self-cond, return is a tuple.
                denoise_inputs = diffusion_utils.add_self_cond_hidden(
                    images_noised, gamma, cond_denoise, num_sc_examples,
                    self.hidden_shapes, drop_rate=sc_drop_rate, training=training)
        else:
            denoise_inputs = images_noised
        cond_denoise = self.get_cond_denoise(labels, for_loss=True)
        denoise_out = cond_denoise(denoise_inputs, gamma, training=training)
        if isinstance(denoise_out, tuple): denoise_out = denoise_out[0]
        return images, noise, images_noised, denoise_out

    def compute_loss(self,
                     images: tf.Tensor,
                     noise: tf.Tensor,
                     denoise_out: tf.Tensor) -> tf.Tensor:
        config = self.config
        if config.pred_type == 'x':
            loss = tf.reduce_mean(tf.square(images - denoise_out))
        elif config.pred_type == 'x_sigmoid_xent':
            pp = tf.nn.sigmoid(denoise_out * images / config.b_scale)
            loss = tf.reduce_mean(-tf.math.log(pp + 1e-8))
        elif config.pred_type == 'x_softmax_xent':
            images = (images / config.b_scale + 1) / 2.  # [-b, b] --> [0, 1]
            images = unfold_rgb(images)
            losses = tf.nn.softmax_cross_entropy_with_logits(images, denoise_out)
            loss = tf.reduce_mean(losses)
        elif config.pred_type == 'eps':
            loss = tf.reduce_mean(tf.square(noise - denoise_out))
        else:
            raise ValueError(f'Unknown pred_type {config.pred_type}')
        return loss

    def call(self,
             images: tf.Tensor,
             labels: tf.Tensor,
             training: bool = True,
             **kwargs) -> tf.Tensor:  # pylint: disable=signature-mismatch
        """Model inference call."""
        with tf.name_scope(''):  # for other functions to have the same name scope
            images, noise, _, denoise_out = self.noise_denoise(
                images, labels, None, training)
            return self.compute_loss(images, noise, denoise_out)
