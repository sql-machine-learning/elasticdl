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
from deepctr.feature_column import DenseFeat, SparseFeat
from deepctr.models import WDL

from elasticdl.python.elasticdl.callbacks import MaxStepsStopping


def custom_model():
    sparse_features = ["C" + str(i) for i in range(1, 27)]
    dense_features = ["I" + str(i) for i in range(1, 14)]
    fixlen_feature_columns = [
        SparseFeat(
            feat,
            vocabulary_size=10000,
            embedding_dim=4,
            dtype="string",
            use_hash=True,
        )
        for i, feat in enumerate(sparse_features)
    ] + [DenseFeat(feat, 1,) for feat in dense_features]

    model = WDL(fixlen_feature_columns, fixlen_feature_columns, task="binary")
    return model


def loss(labels, predictions):
    return tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            y_true=tf.cast(labels, tf.float32), y_pred=predictions,
        )
    )


def optimizer(lr=0.001):
    return tf.keras.optimizers.Adam(learning_rate=lr)


def eval_metrics_fn():
    return {"auc": tf.keras.metrics.AUC()}


def callbacks():
    return [
        MaxStepsStopping(max_steps=150000),
    ]


def feed(dataset, mode, _):
    dataset = dataset.shuffle(10000)
    dataset = dataset.map(parse_data, num_parallel_calls=8)

    return dataset


def parse_data(record):
    feature_description = {"label": tf.io.FixedLenFeature([], tf.int64)}
    for i in range(1, 27):
        feature_description["C" + str(i)] = tf.io.FixedLenFeature(
            (1,), tf.int64
        )

    for i in range(1, 14):
        feature_description["I" + str(i)] = tf.io.FixedLenFeature(
            (1,), tf.int64
        )

    parsed_record = tf.io.parse_single_example(record, feature_description)
    label = tf.cast([parsed_record.pop("label")], tf.dtypes.float32)

    return parsed_record, label


if __name__ == "__main__":
    model = custom_model()

    sparse_features = ["C" + str(i) for i in range(1, 27)]
    dense_features = ["I" + str(i) for i in range(1, 14)]

    test_data = {}
    for name in dense_features:
        test_data[name] = tf.constant([[1.0]])
    for name in sparse_features:
        test_data[name] = tf.constant([["aa"]])

    print(model.call(test_data))
