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

from elasticdl.python.elasticdl.feature_column import feature_column

CATEGORICAL_FEATURE_KEYS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
NUMERIC_FEATURE_KEYS = [
    "age",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
LABEL_KEY = "label"


def get_feature_columns():
    feature_columns = []

    for numeric_feature_key in NUMERIC_FEATURE_KEYS:
        numeric_feature = tf.feature_column.numeric_column(numeric_feature_key)
        feature_columns.append(numeric_feature)

    for categorical_feature_key in CATEGORICAL_FEATURE_KEYS:
        embedding_feature = feature_column.embedding_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                categorical_feature_key, hash_bucket_size=64
            ),
            dimension=16,
        )
        feature_columns.append(embedding_feature)

    return feature_columns


def get_feature_input_layers():
    feature_input_layers = {}

    for numeric_feature_key in NUMERIC_FEATURE_KEYS:
        feature_input_layers[numeric_feature_key] = tf.keras.Input(
            shape=(1,), name=numeric_feature_key, dtype=tf.float32
        )

    for categorical_feature_key in CATEGORICAL_FEATURE_KEYS:
        feature_input_layers[categorical_feature_key] = tf.keras.Input(
            shape=(1,), name=categorical_feature_key, dtype=tf.string
        )

    return feature_input_layers
