# Copyright 2020 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from elasticdl.python.common.constants import Mode


class CustomModel(tf.keras.Model):
    def __init__(self, channel_last=True):
        super(CustomModel, self).__init__(name="mnist_model")
        if channel_last:
            self._reshape = tf.keras.layers.Reshape((28, 28, 1))
        else:
            self._reshape = tf.keras.layers.Reshape((1, 28, 28))
        self._conv1 = tf.keras.layers.Conv2D(
            32, kernel_size=(3, 3), activation="relu"
        )
        self._conv2 = tf.keras.layers.Conv2D(
            64, kernel_size=(3, 3), activation="relu"
        )
        self._batch_norm = tf.keras.layers.BatchNormalization()
        self._maxpooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self._dropout = tf.keras.layers.Dropout(0.25)
        self._flatten = tf.keras.layers.Flatten()
        self._dense = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self._reshape(inputs["image"])
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._batch_norm(x, training=training)
        x = self._maxpooling(x)
        if training:
            x = self._dropout(x, training=training)
        x = self._flatten(x)
        x = self._dense(x)
        return x


def loss(labels, predictions):
    labels = tf.reshape(labels, [-1])
    return tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=predictions, labels=labels
        )
    )


def optimizer(lr=0.01):
    return tf.optimizers.SGD(lr)


def feed(dataset, mode, _):
    def _parse_fn(record):
        return _parse_data(record, mode)

    dataset = dataset.map(_parse_fn)

    if mode == Mode.TRAINING:
        dataset = dataset.shuffle(buffer_size=1024)
    return dataset


def _parse_data(record, mode):
    if mode == Mode.PREDICTION:
        feature_description = {
            "image": tf.io.FixedLenFeature([28, 28], tf.float32)
        }
    else:
        feature_description = {
            "image": tf.io.FixedLenFeature([28, 28], tf.float32),
            "label": tf.io.FixedLenFeature([1], tf.int64),
        }
    r = tf.io.parse_single_example(record, feature_description)
    features = {
        "image": tf.math.divide(tf.cast(r["image"], tf.float32), 255.0)
    }
    if mode == Mode.PREDICTION:
        return features
    else:
        return features, tf.cast(r["label"], tf.int32)


def eval_metrics_fn():
    return {
        "accuracy": lambda labels, predictions: tf.equal(
            tf.argmax(predictions, 1, output_type=tf.int32),
            tf.cast(tf.reshape(labels, [-1]), tf.int32),
        )
    }
