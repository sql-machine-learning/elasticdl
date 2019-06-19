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
        tensor_dict = decode_fn(r, feature_spec)
        label = tensor_dict['y'].numpy().astype(np.int32)
        y_list.append(label)
        feature = tensor_dict['x'].numpy().astype(np.float32)
        x_list.append(feature)

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
