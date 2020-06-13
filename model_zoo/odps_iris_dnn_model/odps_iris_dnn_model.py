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


def custom_model():
    inputs = tf.keras.layers.Input(shape=(4, 1), name="input")
    x = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(3, name="output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="simple-model")


def loss(labels, predictions):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            tf.cast(tf.reshape(labels, [-1]), tf.int32), predictions
        )
    )


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def eval_metrics_fn():
    return {
        "accuracy": lambda labels, predictions: tf.equal(
            tf.argmax(predictions, 1, output_type=tf.int32),
            tf.cast(tf.reshape(labels, [-1]), tf.int32),
        )
    }


def dataset_fn(dataset, mode, metadata):
    def _parse_data(record):
        features = tf.strings.to_number(record[0:-1], tf.float32)
        label = tf.strings.to_number(record[-1], tf.float32)
        return features, label

    dataset = dataset.map(_parse_data)
    return dataset
