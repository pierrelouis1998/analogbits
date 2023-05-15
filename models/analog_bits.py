import tensorflow as tf


def get_model(input_shape : tf.TensorShape):
    """Return tensorflow model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=input_shape),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(input_shape[-1])
    ])
    return model
