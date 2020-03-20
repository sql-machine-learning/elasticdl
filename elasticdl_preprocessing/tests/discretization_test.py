import unittest

import numpy as np
import tensorflow as tf

from elasticdl_preprocessing.layers.discretization import Discretization
from elasticdl_preprocessing.tests.test_utils import (
    ragged_tensor_equal,
    sparse_tensor_equal,
)


class DiscretizationTest(unittest.TestCase):
    def test_discretize(self):
        discretize_layer = Discretization(bins=[1, 5, 10])

        dense_in = tf.constant([[0.2], [1.6], [4.2], [6.1], [10.9]])
        dense_out = discretize_layer(dense_in)
        expected_out = np.array([[0], [1], [1], [2], [3]])
        self.assertTrue(np.array_equal(dense_out.numpy(), expected_out))

        ragged_input = tf.ragged.constant([[0.2, 1.6, 4.2], [6.1], [10.9]])
        ragged_output = discretize_layer(ragged_input)
        expected_ragged_out = tf.ragged.constant(
            [[0, 1, 1], [2], [3]], dtype=tf.int64
        )
        self.assertTrue(
            ragged_tensor_equal(ragged_output, expected_ragged_out)
        )

        sparse_input = ragged_input.to_sparse()
        sparse_output = discretize_layer(sparse_input)
        expected_sparse_out = expected_ragged_out.to_sparse()
        self.assertTrue(
            sparse_tensor_equal(sparse_output, expected_sparse_out)
        )
