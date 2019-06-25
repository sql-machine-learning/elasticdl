import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

inputs = Input(shape=(1, 1), name="x")
outputs = Dense(1)(inputs)
model = Model(inputs, outputs)


def loss(predictions, labels):
    return tf.reduce_mean(tf.square(predictions - labels))


def data_schema():
    return [
        {"name": "x", "shape": [1], "dtype": tf.dtypes.float32},
        {"name": "y", "shape": [1], "dtype": tf.dtypes.float32},
    ]


def input_fn(records):
    x_list = []
    y_list = []
    # deserialize
    for r in records:
        x_list.append([r["x"]])
        y_list.append([r["y"]])
    # batching
    batch_size = len(x_list)
    xs = np.concatenate(x_list, axis=0)
    xs = np.reshape(xs, (batch_size, 1))
    ys = np.reshape(xs, (batch_size, 1))
    xs = tf.convert_to_tensor(xs)
    return {"x": xs}, ys


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def eval_metrics_fn(predictions, labels):
    return {"mse": tf.reduce_mean(tf.square(predictions - labels))}
