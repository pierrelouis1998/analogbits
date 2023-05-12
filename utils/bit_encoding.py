"""Bit encoding utils"""
import tensorflow as tf


class GrayCode:
    """Gray code converter."""

    def __init__(self, n_bits):
        self._n = n_bits
        self.codes = tf.constant(self.gen_gray_codes(n_bits), dtype=tf.int32)
        self.inv_codes = tf.math.invert_permutation(self.codes)

    def to_gray_code_tensor(self, x: tf.Tensor) -> tf.Tensor:
        return tf.gather(self.codes, tf.cast(x, tf.int32))

    def from_gray_code_tensor(self, x: tf.Tensor) -> tf.Tensor:
        return tf.gather(self.inv_codes, tf.cast(x, tf.int32))

    def gen_gray_codes(self, n):
        assert n > 0
        if n == 1:
            return [0, 1]
        shorter_gray_codes = self.gen_gray_codes(n - 1)
        bitmask = 1 << (n - 1)
        gray_codes = list(shorter_gray_codes)
        for gray_code in reversed(shorter_gray_codes):
            gray_codes.append(bitmask | gray_code)
        return gray_codes


def rgb2bit(images, b_type, b_scale, x_channels):  # pylint: disable=missing-function-docstring
    """

    :param images: Shape (bsize, h, w, channels) or (h, w, channels)
    :param b_type:
    :param b_scale:
    :param x_channels: Value obtained by get_x_channels
    :return:
    """
    if b_type in ['uint8', 'uint8_s', 'gray']:
        images = tf.image.convert_image_dtype(images, dtype=tf.uint8)
        if b_type in ['uint8_s', 'gray']:
            images = tf.gather(get_perm_inv_perm(b_type)[0],
                               tf.cast(images, tf.int32))
        images = int2bits(
            tf.cast(images, tf.int32), x_channels // 3, tf.float32)
        sh = images.shape
        images = tf.reshape(images, sh[:-2] + [sh[-2] * sh[-1]])
        images = (images * 2 - 1) * b_scale
    elif b_type == 'oneh':
        images = tf.image.convert_image_dtype(images, dtype=tf.uint8)
        images = tf.one_hot(images, 256)
        sh = images.shape
        images = tf.reshape(images, sh[:-2] + [sh[-2] * sh[-1]])
        images = (images * 2 - 1) * b_scale
    else:
        raise ValueError(f'Unknown b_type {b_type}')
    return images


def int2bits(x, n, out_dtype=None):
    """Convert an integer x in (...) into bits in (..., n)."""
    x = tf.bitwise.right_shift(tf.expand_dims(x, -1), tf.range(n))
    x = tf.math.mod(x, 2)
    if out_dtype and out_dtype != x.dtype:
        x = tf.cast(x, out_dtype)
    return x


def bits2int(x, out_dtype):
    """Converts bits x in (..., n) into an integer in (...)."""
    x = tf.cast(x, out_dtype)
    x = tf.math.reduce_sum(x * (2 ** tf.range(tf.shape(x)[-1])), -1)
    return x


def get_perm_inv_perm(b_type: str):
    if b_type == 'uint8_s':
        # perm/inv_perm generation
        # np.random.seed(42)
        # perm = np.arange(256)
        # np.random.shuffle(perm)
        # perm = tf.constant(perm, dtype=tf.int32)
        # inv_perm = tf.math.invert_permutation(perm)
        perm = tf.constant(
            [228, 6, 79, 206, 117, 185, 242, 167, 9, 30, 180, 222, 230,
             217, 136, 68, 199, 15, 96, 24, 235, 19, 120, 152, 33, 124,
             253, 208, 10, 164, 184, 97, 148, 190, 223, 25, 86, 18, 75,
             137, 196, 176, 239, 181, 45, 66, 16, 67, 215, 201, 177, 38,
             143, 84, 55, 220, 104, 139, 127, 60, 101, 172, 245, 126, 225,
             144, 108, 178, 73, 114, 158, 69, 141, 109, 115, 246, 113, 243,
             90, 29, 170, 82, 111, 5, 56, 132, 154, 162, 65, 186, 85,
             219, 237, 31, 12, 35, 28, 42, 112, 22, 125, 93, 173, 251,
             51, 240, 95, 146, 204, 76, 41, 119, 155, 78, 150, 26, 247,
             168, 118, 193, 140, 0, 2, 77, 46, 100, 205, 159, 183, 254,
             98, 36, 61, 200, 142, 11, 250, 224, 27, 231, 4, 122, 32,
             147, 182, 138, 62, 135, 128, 232, 194, 70, 197, 64, 44, 165,
             156, 40, 123, 153, 23, 192, 249, 81, 39, 244, 47, 94, 195,
             161, 43, 145, 175, 3, 105, 53, 133, 233, 198, 238, 49, 163,
             80, 34, 211, 7, 171, 216, 110, 91, 83, 229, 234, 89, 8,
             13, 59, 221, 131, 17, 166, 72, 226, 134, 209, 236, 63, 54,
             107, 50, 212, 174, 213, 189, 252, 207, 227, 169, 58, 218, 48,
             88, 21, 57, 203, 160, 248, 187, 191, 129, 37, 157, 241, 1,
             52, 149, 130, 151, 103, 99, 116, 87, 202, 74, 214, 210, 121,
             255, 20, 188, 71, 106, 14, 92, 179, 102], dtype=tf.int32)

        inv_perm = tf.constant(
            [121, 233, 122, 173, 140, 83, 1, 185, 194, 8, 28, 135, 94,
             195, 252, 17, 46, 199, 37, 21, 248, 222, 99, 160, 19, 35,
             115, 138, 96, 79, 9, 93, 142, 24, 183, 95, 131, 230, 51,
             164, 157, 110, 97, 170, 154, 44, 124, 166, 220, 180, 209, 104,
             234, 175, 207, 54, 84, 223, 218, 196, 59, 132, 146, 206, 153,
             88, 45, 47, 15, 71, 151, 250, 201, 68, 243, 38, 109, 123,
             113, 2, 182, 163, 81, 190, 53, 90, 36, 241, 221, 193, 78,
             189, 253, 101, 167, 106, 18, 31, 130, 239, 125, 60, 255, 238,
             56, 174, 251, 208, 66, 73, 188, 82, 98, 76, 69, 74, 240,
             4, 118, 111, 22, 246, 141, 158, 25, 100, 63, 58, 148, 229,
             236, 198, 85, 176, 203, 147, 14, 39, 145, 57, 120, 72, 134,
             52, 65, 171, 107, 143, 32, 235, 114, 237, 23, 159, 86, 112,
             156, 231, 70, 127, 225, 169, 87, 181, 29, 155, 200, 7, 117,
             217, 80, 186, 61, 102, 211, 172, 41, 50, 67, 254, 10, 43,
             144, 128, 30, 5, 89, 227, 249, 213, 33, 228, 161, 119, 150,
             168, 40, 152, 178, 16, 133, 49, 242, 224, 108, 126, 3, 215,
             27, 204, 245, 184, 210, 212, 244, 48, 187, 13, 219, 91, 55,
             197, 11, 34, 137, 64, 202, 216, 0, 191, 12, 139, 149, 177,
             192, 20, 205, 92, 179, 42, 105, 232, 6, 77, 165, 62, 75,
             116, 226, 162, 136, 103, 214, 26, 129, 247], dtype=tf.int32)
        return perm, inv_perm
    elif b_type == 'gray':
        gray_code = GrayCode(8)
        return gray_code.codes, gray_code.inv_codes
    else:
        raise ValueError(f'Unknown b_type {b_type}')


# pylint: enable=bad-whitespace,bad-continuation

def get_x_channels(b_type):
    x_channels = 24 if b_type in ['uint8', 'uint8_s', 'gray'] else 9
    if b_type == 'oneh':
        x_channels = 256 * 3
    return x_channels


def bit2rgb(samples, b_type):  # pylint: disable=missing-function-docstring
    if b_type in ['uint8', 'uint8_s', 'gray']:
        samples = unfold_rgb(samples)
        samples = bits2int(samples > 0, tf.int32)
        if b_type in ['uint8_s', 'gray']:
            samples = tf.gather(get_perm_inv_perm(b_type)[1], samples)
        return tf.image.convert_image_dtype(
            tf.cast(samples, tf.uint8), dtype=tf.float32)
    elif b_type == 'oneh':
        samples = unfold_rgb(samples)
        samples = tf.argmax(samples, -1)
        return tf.image.convert_image_dtype(
            tf.cast(samples, tf.uint8), dtype=tf.float32)
    else:
        raise ValueError(f'Unknown b_type {b_type}')


def unfold_rgb(x):
    # (b, h, w, d=3k) --> (b, h, w, 3, k) or (h, w, d=3k) --> (h, w, 3, k)
    if x.shape.rank <= 4:
        sh = shape_as_list(x)
        x = tf.reshape(x, sh[:-1] + [3, sh[-1] // 3])
    return x


def fold_rgb(x):
    # (b, h, w, 3, k) --> (b, h, w, d=3k)
    if x.shape.rank == 5:
        sh = shape_as_list(x)
        x = tf.reshape(x, sh[:-2] + [sh[-1] * sh[-2]])
    return x


def shape_as_list(t):
    # Assumes rank of `t` is statically known.
    shape = t.shape.as_list()
    dynamic_shape = tf.shape(t)
    return [
        shape[i] if shape[i] is not None else dynamic_shape[i]
        for i in range(len(shape))
    ]
