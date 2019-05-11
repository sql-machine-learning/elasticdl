import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import *

class TestModel(tf.keras.Model):
    def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        return self.dense(inputs)

inputs = Input(shape=(1, 1))
outputs = Dense(1)(inputs)
model = tf.keras.Model(inputs, outputs)

input_names = ['x']

def loss(outputs, labels):
    return tf.reduce_mean(tf.square(outputs - labels['y'])) 

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
    return {'x': xs, 'y': ys}

def optimizer(lr=0.1):
    return tf.train.GradientDescentOptimizer(lr)
