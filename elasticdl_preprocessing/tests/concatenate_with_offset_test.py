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

from elasticdl_preprocessing.layers.concatenate_with_offset import (
    ConcatenateWithOffset,
)
from elasticdl_preprocessing.tests.test_utils import (
    ragged_tensor_equal,
    sparse_tensor_equal,
)


class ConcatenateWithOffsetTest(unittest.TestCase):
    def test_concatenate_with_offset(self):
        tensor_1 = tf.constant([[1], [1], [1]])
        tensor_2 = tf.constant([[2], [2], [2]])
        offsets = [0, 10]
        concat_layer = ConcatenateWithOffset(offsets=offsets, axis=1)

        output = concat_layer([tensor_1, tensor_2])
        expected_out = np.array([[1, 12], [1, 12], [1, 12]])
        self.assertTrue(np.array_equal(output.numpy(), expected_out))

        ragged_tensor_1 = tf.ragged.constant([[1], [], [1]], dtype=tf.int64)
        ragged_tensor_2 = tf.ragged.constant([[2], [2], []], dtype=tf.int64)
        output = concat_layer([ragged_tensor_1, ragged_tensor_2])
        expected_out = tf.ragged.constant([[1, 12], [12], [1]], dtype=tf.int64)
        self.assertTrue(ragged_tensor_equal(output, expected_out))

        sparse_tensor_1 = ragged_tensor_1.to_sparse()
        sparse_tensor_2 = ragged_tensor_2.to_sparse()
        output = concat_layer([sparse_tensor_1, sparse_tensor_2])
        expected_out = tf.SparseTensor(
            indices=np.array([[0, 0], [0, 1], [1, 1], [2, 0]]),
            values=np.array([1, 12, 12, 1]),
            dense_shape=(3, 2),
        )
        self.assertTrue(sparse_tensor_equal(output, expected_out))
