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
from elasticdl.python.elasticdl.callbacks import LearningRateScheduler
from model_zoo.cifar10.data_parser import parse_data


def custom_model():
    input_image = tf.keras.layers.Input(shape=(32, 32, 3), name="image")
    model = tf.keras.applications.ResNet50(
        include_top=True,
        weights=None,
        input_tensor=input_image,
        input_shape=None,
        pooling=None,
        classes=10,
    )
    return model


def loss(labels, predictions):
    labels = tf.reshape(labels, [-1])
    return tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=predictions, labels=labels
        )
    )


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def callbacks():
    def _schedule(model_version):
        if model_version < 5000:
            return 0.1
        elif model_version < 15000:
            return 0.01
        else:
            return 0.001

    return [LearningRateScheduler(_schedule)]


def dataset_fn(dataset, mode, _):
    def _parse_data(record):
        return parse_data(record, mode)

    dataset = dataset.map(_parse_data)

    if mode == Mode.TRAINING:
        dataset = dataset.shuffle(buffer_size=1024)
    return dataset


def eval_metrics_fn():
    return {
        "accuracy": lambda labels, predictions: tf.equal(
            tf.argmax(predictions, 1, output_type=tf.int32),
            tf.cast(tf.reshape(labels, [-1]), tf.int32),
        )
    }
