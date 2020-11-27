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

from model_zoo.dac_ctr.feature_config import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    HASH_FEATURES,
    LABEL_KEY,
    STANDARDIZED_FEATURES,
)
from model_zoo.dac_ctr.feature_transform import transform_feature
from model_zoo.dac_ctr.xdeepfm_model import xdeepfm_model as ctr_model


def custom_model():
    # The codes in the method can all be auto-generated
    input_layers = get_input_layers(FEATURE_NAMES)
    standardized_tensor, id_tensors, max_ids = transform_feature(
        input_layers, feature_groups=FEATURE_GROUPS
    )
    model = ctr_model(input_layers, standardized_tensor, id_tensors, max_ids,)
    return model


def loss(labels, predictions):
    logits = predictions["logits"]
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(tf.reshape(labels, (-1, 1)), tf.float32),
            logits=logits,
        )
    )


def optimizer(lr=0.001):
    return tf.keras.optimizers.Adam(learning_rate=lr)


def eval_metrics_fn():
    return {
        "logits": {
            "accuracy": lambda labels, predictions: tf.equal(
                tf.cast(tf.reshape(predictions, [-1]) > 0.5, tf.int32),
                tf.cast(tf.reshape(labels, [-1]), tf.int32),
            )
        },
        "probs": {"auc": tf.keras.metrics.AUC()},
    }


def feed(dataset, mode, _):
    def _parse_data(record):
        feature_description = dict(
            [
                (name, tf.io.FixedLenFeature((1,), tf.int64))
                for name in STANDARDIZED_FEATURES
            ]
            + [
                (name, tf.io.FixedLenFeature((1,), tf.string))
                for name in HASH_FEATURES
            ]
            + [(LABEL_KEY, tf.io.FixedLenFeature([], tf.int64))]
        )

        parsed_record = tf.io.parse_single_example(record, feature_description)
        label = parsed_record.pop(LABEL_KEY)

        return parsed_record, label

    dataset = dataset.prefetch(10000)
    dataset = dataset.shuffle(10000)
    dataset = dataset.map(_parse_data, num_parallel_calls=8)

    return dataset


def get_input_layers(feature_names):
    input_layers = {}
    for name in feature_names:
        if name in STANDARDIZED_FEATURES:
            dtype = tf.int64
        else:
            dtype = tf.string
        input_layers[name] = tf.keras.layers.Input(
            name=name, shape=(1,), dtype=dtype
        )

    return input_layers


if __name__ == "__main__":
    model = custom_model()
    test_data = {}
    for name in STANDARDIZED_FEATURES:
        test_data[name] = tf.constant([[10]])
    for name in HASH_FEATURES:
        test_data[name] = tf.constant([["aa"]])
    print(model.call(test_data))
