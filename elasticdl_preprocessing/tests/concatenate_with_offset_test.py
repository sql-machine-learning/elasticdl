import unittest

import numpy as np
import tensorflow as tf

from elasticdl_preprocessing.layers.concatenate_with_offset import (
    ConcatenateWithOffset
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

        ragged_tensor_1 = tf.ragged.constant([[1], [], [1]])
        ragged_tensor_2 = tf.ragged.constant([[2], [2], []])
        output = concat_layer([ragged_tensor_1, ragged_tensor_2])
        expected_out = np.array([1, 12, 12, 1])
        self.assertTrue(np.array_equal(output.values.numpy(), expected_out))

        sparse_tensor_1 = ragged_tensor_1.to_sparse()
        sparse_tensor_2 = ragged_tensor_2.to_sparse()
        output = concat_layer([sparse_tensor_1, sparse_tensor_2])
        self.assertTrue(np.array_equal(output.values.numpy(), expected_out))
