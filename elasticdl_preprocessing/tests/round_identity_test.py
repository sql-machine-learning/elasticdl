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

from elasticdl_preprocessing.layers.round_identity import RoundIdentity
from elasticdl_preprocessing.tests.test_utils import (
    ragged_tensor_equal,
    sparse_tensor_equal,
)


class RoundIdentityTest(unittest.TestCase):
    def test_round_indentity(self):
        round_identity = RoundIdentity(num_buckets=10)

        dense_input = tf.constant([[1.2], [1.6], [0.2], [3.1], [4.9]])
        output = round_identity(dense_input)
        expected_out = np.array([[1], [2], [0], [3], [5]])
        self.assertTrue(np.array_equal(output.numpy(), expected_out))

        ragged_input = tf.ragged.constant([[1.1, 3.4], [0.5]])
        ragged_output = round_identity(ragged_input)
        expected_ragged_out = tf.ragged.constant([[1, 3], [0]], dtype=tf.int64)
        self.assertTrue(
            ragged_tensor_equal(ragged_output, expected_ragged_out)
        )

        sparse_input = ragged_input.to_sparse()
        sparse_output = round_identity(sparse_input)
        expected_sparse_out = expected_ragged_out.to_sparse()
        self.assertTrue(
            sparse_tensor_equal(sparse_output, expected_sparse_out)
        )
