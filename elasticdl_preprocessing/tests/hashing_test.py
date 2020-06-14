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

from elasticdl_preprocessing.layers.hashing import Hashing
from elasticdl_preprocessing.tests.test_utils import (
    ragged_tensor_equal,
    sparse_tensor_equal,
)


class HashingTest(unittest.TestCase):
    def test_hashing(self):
        hash_layer = Hashing(num_bins=3)
        inp = np.asarray([["A"], ["B"], ["C"], ["D"], ["E"]])
        hash_out = hash_layer(inp)
        expected_out = np.array([[1], [0], [1], [1], [2]])
        self.assertTrue(np.array_equal(hash_out.numpy(), expected_out))

        ragged_in = tf.ragged.constant([["A", "B"], ["C", "D"], ["E"], []])
        hash_out = hash_layer(ragged_in)
        expected_ragged_out = tf.ragged.constant(
            [[1, 0], [1, 1], [2], []], dtype=tf.int64
        )
        self.assertTrue(ragged_tensor_equal(hash_out, expected_ragged_out))

        sparse_in = ragged_in.to_sparse()
        hash_out = hash_layer(sparse_in)
        expected_sparse_out = expected_ragged_out.to_sparse()
        self.assertTrue(sparse_tensor_equal(hash_out, expected_sparse_out))
