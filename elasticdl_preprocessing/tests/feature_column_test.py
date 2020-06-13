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

import copy
import unittest

import numpy as np
import tensorflow as tf

from elasticdl_preprocessing.feature_column.feature_column import (
    concatenated_categorical_column,
)


def call_feature_columns(feature_columns, input):
    dense_features = tf.keras.layers.DenseFeatures(feature_columns)
    return dense_features(input)


class ConcatenatedCategoricalColumnTest(unittest.TestCase):
    def test_name(self):
        a = tf.feature_column.categorical_column_with_hash_bucket(
            "aaa", hash_bucket_size=1024
        )
        b = tf.feature_column.categorical_column_with_identity(
            "bbb", num_buckets=32
        )
        c = tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column("ccc"), boundaries=[1, 2, 3, 4, 5]
        )
        concat = concatenated_categorical_column([a, b, c])
        self.assertEqual("aaa_C_bbb_C_ccc_bucketized", concat.name)

    def test_is_v2_column(self):
        a = tf.feature_column.categorical_column_with_hash_bucket(
            "aaa", hash_bucket_size=1024
        )
        b = tf.feature_column.categorical_column_with_identity(
            "bbb", num_buckets=32
        )
        concat = concatenated_categorical_column([a, b])
        self.assertTrue(concat._is_v2_column)

    def test_num_buckets(self):
        a = tf.feature_column.categorical_column_with_hash_bucket(
            "aaa", hash_bucket_size=1024
        )
        b = tf.feature_column.categorical_column_with_identity(
            "bbb", num_buckets=32
        )
        concat = concatenated_categorical_column([a, b])
        self.assertEqual(1056, concat.num_buckets)

    def test_parse_spec(self):
        a = tf.feature_column.categorical_column_with_hash_bucket(
            "aaa", hash_bucket_size=1024, dtype=tf.string
        )
        b = tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column("bbb", dtype=tf.int32),
            boundaries=[1, 2, 3, 4, 5],
        )
        concat = concatenated_categorical_column([a, b])
        self.assertEqual(
            {
                "aaa": tf.io.VarLenFeature(dtype=tf.string),
                "bbb": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int32),
            },
            concat.parse_example_spec,
        )

    def test_deep_copy(self):
        a = tf.feature_column.categorical_column_with_hash_bucket(
            "aaa", hash_bucket_size=1024
        )
        b = tf.feature_column.categorical_column_with_identity(
            "bbb", num_buckets=32
        )
        concat = concatenated_categorical_column([a, b])
        concat_copy = copy.deepcopy(concat)
        self.assertEqual("aaa_C_bbb", concat_copy.name)
        self.assertEqual(1056, concat_copy.num_buckets)

    def test_call_column(self):
        user_id = tf.feature_column.categorical_column_with_identity(
            "user_id", num_buckets=32
        )

        item_id = tf.feature_column.categorical_column_with_identity(
            "item_id", num_buckets=128
        )

        item_id_user_id_concat = concatenated_categorical_column(
            [user_id, item_id]
        )

        concat_indicator = tf.feature_column.indicator_column(
            item_id_user_id_concat
        )

        output = call_feature_columns(
            [concat_indicator], {"user_id": [10, 20], "item_id": [1, 120]},
        )

        expected_output = tf.one_hot(indices=[10, 20], depth=160) + tf.one_hot(
            indices=[1 + 32, 120 + 32], depth=160
        )

        self.assertTrue(
            np.array_equal(output.numpy(), expected_output.numpy())
        )


if __name__ == "__main__":
    unittest.main()
