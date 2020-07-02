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

import unittest

import tensorflow as tf

from elasticdl_preprocessing.layers.to_ragged import ToRagged
from elasticdl_preprocessing.tests.test_utils import ragged_tensor_equal


class ToRaggedTest(unittest.TestCase):
    def test_dense_to_ragged(self):
        layer = ToRagged()
        input_data = tf.constant([[1], [-1], [4]], tf.int64)
        out = layer(input_data)
        expected_out = tf.ragged.constant([[1], [], [4]], tf.int64)
        self.assertTrue(ragged_tensor_equal(out, expected_out))

    def test_string_split_to_ragged(self):
        layer = ToRagged()
        input_data = tf.ragged.constant([["1", "2", "3"], ["4", "5"], [""]])
        out = layer(input_data)
        expected_out = tf.ragged.constant([["1", "2", "3"], ["4", "5"], []])
        self.assertTrue(ragged_tensor_equal(out, expected_out))

    def test_model_with_ragged(self):
        inputs = tf.keras.Input(shape=(1,), dtype=tf.int32)
        ragged = ToRagged(ignore_value=-1)(inputs)
        sum_out = tf.reduce_sum(ragged)
        model = tf.keras.Model(inputs=inputs, outputs=sum_out)
        sum_value = model.call(tf.constant([[1], [-1], [4]])).numpy()
        self.assertEqual(sum_value, 5.0)


if __name__ == "__main__":
    unittest.main()
