import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

inputs = Input(shape=(1, 1))
outputs = Dense(1)(inputs)
model = Model(inputs, outputs)


def loss(predictions, labels):
    return tf.reduce_mean(tf.square(predictions - labels))


feature_label_colums = [
    tf.feature_column.numeric_column(
        key="x", dtype=tf.dtypes.float32, shape=[1]
    ),
    tf.feature_column.numeric_column(
        key="y", dtype=tf.dtypes.float32, shape=[1]
    ),
]
feature_spec = tf.feature_column.make_parse_example_spec(feature_label_colums)


def input_fn(records, decode_fn):
    x_list = []
    y_list = []
    # deserialize
    for r in records:
        example = decode_fn(r)

        x = np.asarray(example.features.feature['x'].float_list.value)
        x_list.append(x.astype(np.float32))

        y = np.asarray(example.features.feature['y'].float_list.value)
        y_list.append(y[0])

    # batching
    batch_size = len(x_list)
    xs = np.concatenate(x_list, axis=0)
    xs = np.reshape(xs, (batch_size, 1))
    xs = tf.convert_to_tensor(xs)
    ys = np.array(y_list)
    return [xs], ys


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def eval_metrics_fn(predictions, labels):
    return {"mse": tf.reduce_mean(tf.square(predictions - labels))}
