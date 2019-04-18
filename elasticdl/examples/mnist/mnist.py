import tensorflow as tf
tf.enable_eager_execution()

import os
import argparse
import numpy as np
from worker.worker import Worker


class MnistModel(tf.keras.Model):
    def __init__(self, channel_last=True):
        super(MnistModel, self).__init__(name='mnist_model')
        if channel_last:
            self._reshape = tf.keras.layers.Reshape((28, 28, 1))
        else:
            self._reshape = tf.keras.layers.Reshape((1, 28, 28))
        self._conv1 = tf.keras.layers.Conv2D(
            32, kernel_size=(3, 3), activation='relu')
        self._conv2 = tf.keras.layers.Conv2D(
            64, kernel_size=(3, 3), activation='relu')
        self._batch_norm = tf.keras.layers.BatchNormalization()
        self._maxpooling = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2))
        self._dropout = tf.keras.layers.Dropout(0.25)
        self._flatten = tf.keras.layers.Flatten()
        self._dense = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self._reshape(inputs)
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._batch_norm(x, training=training)
        x = self._maxpooling(x)
        if training:
            x = self._dropout(x, training=training)
        x = self._flatten(x)
        x = self._dense(x)
        return x

    @staticmethod
    def input_shapes():
        return (1, 28, 28)

    @staticmethod
    def input_names():
        return ['image']

    @staticmethod
    def loss(output, labels):
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=output, labels=labels['label']))

    @staticmethod
    def optimizer(lr=0.1):
        return tf.train.GradientDescentOptimizer(lr)

    @staticmethod
    def input_fn(records):
        image_list = []
        label_list = []
        # deserialize
        for r in records:
            parsed = np.frombuffer(r, dtype="uint8")
            label = parsed[-1]
            image = np.resize(parsed[:-1], new_shape=(28, 28))
            image = image.astype(np.float32)
            image /= 255
            label = label.astype(np.int32)
            image_list.append(image)
            label_list.append(label)

        # batching
        batch_size = len(image_list)
        images = np.concatenate(image_list, axis=0)
        images = np.reshape(images, (batch_size, 28, 28))
        labels = np.array(label_list)
        return {'image': images, 'label': labels}
