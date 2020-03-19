import unittest

import numpy as np
import tensorflow as tf

from elasticdl_preprocessing.layers.hashing import Hashing
from elasticdl_preprocessing.tests.test_utils import (
    sparse_tensor_equal,
    ragged_tensor_equal,
)


class HashingTest(unittest.TestCase):
    def test_hashing(self):
        hash_layer = Hashing(num_bins=3)
        inp = np.asarray([['A'], ['B'], ['C'], ['D'], ['E']])
        hash_out = hash_layer(inp)
        expected_out = np.array([[1], [0], [1], [1], [2]])
        self.assertTrue(np.array_equal(hash_out.numpy(), expected_out))

        ragged_in = tf.ragged.constant(
            [['A', 'B'], ['C', 'D'], ['E'], []]
        )
        hash_out = hash_layer(ragged_in)
        expected_ragged_out = tf.ragged.constant(
            [[1, 0], [1, 1], [2], []], dtype=tf.int64
        )
        self.assertTrue(ragged_tensor_equal(hash_out, expected_ragged_out))

        sparse_in = ragged_in.to_sparse()
        hash_out = hash_layer(sparse_in)
        expected_sparse_out = expected_ragged_out.to_sparse()
        self.assertTrue(sparse_tensor_equal(hash_out, expected_sparse_out))
