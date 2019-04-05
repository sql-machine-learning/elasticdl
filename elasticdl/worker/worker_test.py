import unittest
import numpy as np


import tensorflow as tf
tf.enable_eager_execution()

from .worker import Worker


def input_fn(kwargs):
    def gen():
        for i in range(64):
            x = np.random.rand((1)).astype(np.float32)
            y = np.float32(2 * x + 1)
            yield {'x': x, 'y': y}

    dataset = tf.data.Dataset.from_generator(
        gen, output_types={'x': tf.float32, 'y': tf.float32},
        output_shapes={'x': tf.TensorShape([1]), 'y': tf.TensorShape([1])})

    return dataset


def get_optimizer(lr=0.1):
    return tf.train.GradientDescentOptimizer(lr)


class TestModel(object):
    def __init__(self):
        input1 = tf.keras.layers.Input(shape=(1,))
        x1 = tf.keras.layers.Dense(1)(input1)
        self._model = tf.keras.models.Model(input1, x1)

    def get_keras_model(self):
        return self._model

    def output(self, data):
        return self._model.call(data['x'])

    def loss(self, output, data):
        return tf.reduce_mean(tf.square(output - data['y']))


class WorkerTest(unittest.TestCase):
    def test_local_train(self):
        worker = Worker(TestModel, input_fn, get_optimizer)
        batch_size = 32
        epoch = 2
        try:
            worker.local_train(batch_size, epoch)
            res = True
        except Exception as ex:
            print(ex)
            res = False
        self.assertTrue(res)
