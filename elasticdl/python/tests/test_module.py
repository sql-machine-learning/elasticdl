import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import *
from tensorflow.keras import Model

inputs = Input(shape=(1, 1))
outputs = Dense(1)(inputs)
model = Model(inputs, outputs)


def loss(predictions, labels):
    return tf.reduce_mean(tf.square(predictions - labels)) 


def feature_columns():
    return [tf.feature_column.numeric_column(
        key="x", dtype=tf.dtypes.float32, shape=[1])]


def label_columns():
    return [tf.feature_column.numeric_column(
        key="y", dtype=tf.dtypes.float32, shape=[1])]


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
    return {'x': xs}, ys

def optimizer(lr=0.1):
    return tf.train.GradientDescentOptimizer(lr)
