import unittest

import numpy as np
import tensorflow as tf

from elasticdl_preprocessing.layers import LogRound
from elasticdl_preprocessing.tests.test_utils import (
    ragged_tensor_equal,
    sparse_tensor_equal,
)


class LogRoundTest(unittest.TestCase):
    def test_round_indentity(self):
        log_round = LogRound(num_bins=20, base=2)

        dense_input = tf.constant([[1.2], [2.6], [0.2], [3.1], [1024]])
        output = log_round(dense_input)
        expected_out = np.array([[0], [1], [0], [2], [10]])

        self.assertTrue(np.array_equal(output.numpy(), expected_out))

        ragged_input = tf.ragged.constant([[1.1, 3.4], [1025]])
        ragged_output = log_round(ragged_input)
        expected_ragged_out = tf.ragged.constant(
            [[0, 2], [10]], dtype=tf.int64
        )
        self.assertTrue(
            ragged_tensor_equal(ragged_output, expected_ragged_out)
        )

        sparse_input = ragged_input.to_sparse()
        sparse_output = log_round(sparse_input)
        expected_sparse_out = expected_ragged_out.to_sparse()
        self.assertTrue(
            sparse_tensor_equal(sparse_output, expected_sparse_out)
        )

    def test_model_with_loground(self):
        inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
        log_round = LogRound(num_bins=20, base=2)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=log_round)
        out = model.call(tf.constant([[1.2], [2.6], [0.2], [3.1], [1024]]))
        self.assertTrue(
            np.array_equal(
                np.array([[0], [1], [0], [2], [10]], dtype=int), out.numpy()
            )
        )
