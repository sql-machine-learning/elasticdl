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
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input

from elasticdl.python.data.reader.recordio_reader import RecordIODataReader
from elasticdl.python.worker.prediction_outputs_processor import (
    BasePredictionOutputsProcessor,
)


def custom_model():
    inputs = Input(shape=(1, 1), name="x")
    outputs = Dense(1)(inputs)
    return Model(inputs, outputs)


def loss(labels, predictions):
    return tf.reduce_mean(tf.square(predictions - labels))


def keras_loss(labels, predictions):
    return tf.keras.losses.mean_squared_error(labels, predictions)


def dataset_fn(dataset, mode, metadata):
    def _parse_data(record):
        feature_description = {
            "x": tf.io.FixedLenFeature([1], tf.float32),
            "y": tf.io.FixedLenFeature([1], tf.float32),
        }
        r = tf.io.parse_single_example(record, feature_description)
        return {"x": r["x"]}, r["y"]

    dataset = dataset.map(_parse_data)
    return dataset


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def ftrl_optimizer(lr=0.1):
    return tf.optimizers.Ftrl(lr)


def eval_metrics_fn():
    return {"mse": lambda labels, outputs: tf.square(outputs - labels)}


def callbacks():
    return [tf.keras.callbacks.Callback()]


class PredictionOutputsProcessor(BasePredictionOutputsProcessor):
    def __init__(self):
        pass

    def process(self, predictions, worker_id):
        pass


class CustomDataReader(RecordIODataReader):
    def __init__(self, **kwargs):
        RecordIODataReader.__init__(self, **kwargs)

    def custom_method(self):
        return "custom_method"


def custom_data_reader(data_origin, records_per_task=None, **kwargs):
    return CustomDataReader(data_dir=data_origin)
