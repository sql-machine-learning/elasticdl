import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import *
from tensorflow.keras import Model

inputs = Input(shape=(1, 1))
outputs = Dense(1)(inputs)
model = Model(inputs, outputs)

input_names = ['x']

def loss(outputs, labels):
    return tf.reduce_mean(tf.square(outputs - labels)) 

def input_fn(records):
    x_list = []
    y_list = []
    # deserialize
    for r in records:
        parsed = np.frombuffer(r, dtype='float32')
        x_list.append([parsed[0]])
        y_list.append([parsed[1]])
    # batching
    batch_size = len(x_list)
    xs = np.concatenate(x_list, axis=0)
    xs = np.reshape(xs, (batch_size, 1))
    ys = np.concatenate(y_list, axis=0)
    ys = np.reshape(xs, (batch_size, 1))
    return ({'x': xs}, ys)

def optimizer(lr=0.1):
    return tf.train.GradientDescentOptimizer(lr)
