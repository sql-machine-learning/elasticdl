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

import numpy as np
import tensorflow as tf

from elasticdl_preprocessing.layers import Normalizer
from elasticdl_preprocessing.tests.test_utils import (
    ragged_tensor_equal,
    sparse_tensor_equal,
)


class NormalizerTest(unittest.TestCase):
    def test_normalizer(self):
        normalizer = Normalizer(1.0, 2.0)

        dense_input = tf.constant([[5.0], [7.0], [9.0], [11.0], [13.0]])
        output = normalizer(dense_input)
        expected_out = np.array([[2.0], [3.0], [4.0], [5.0], [6.0]])

        self.assertTrue(np.array_equal(output.numpy(), expected_out))

        ragged_input = tf.ragged.constant([[5.0, 7.0], [9.0]])
        ragged_output = normalizer(ragged_input)
        expected_ragged_out = tf.ragged.constant(
            [[2.0, 3.0], [4.0]], dtype=tf.float32
        )
        self.assertTrue(
            ragged_tensor_equal(ragged_output, expected_ragged_out)
        )

        sparse_input = ragged_input.to_sparse()
        sparse_output = normalizer(sparse_input)
        expected_sparse_out = expected_ragged_out.to_sparse()
        self.assertTrue(
            sparse_tensor_equal(sparse_output, expected_sparse_out)
        )

    def test_model_with_normalizer(self):
        inputs = tf.keras.Input(shape=(1,), dtype=tf.float32)
        normalize = Normalizer(1.0, 2.0)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=normalize)
        out = model.call(tf.constant([[5.0], [7.0], [9.0], [11.0], [13.0]]))
        self.assertTrue(
            np.array_equal(
                np.array([[2.0], [3.0], [4.0], [5.0], [6.0]]), out.numpy()
            )
        )
