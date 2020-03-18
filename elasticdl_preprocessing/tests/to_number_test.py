import unittest

import numpy as np
import tensorflow as tf

from elasticdl_preprocessing.layers.to_number import ToNumber
from elasticdl_preprocessing.tests.test_utils import sparse_tensor_equal


class ToNumberTest(unittest.TestCase):
    def test_call_dense(self):
        layer = ToNumber(out_type=tf.int32, default_value=-1)
        input = tf.constant([["123", ""], ["456", "-789"]], tf.string)
        output = layer.call(input)
        expected_output = tf.constant([[123, -1], [456, -789]], tf.int32)
        self.assertEqual(output.dtype, tf.int32)
        self.assertTrue(
            np.array_equal(output.numpy(), expected_output.numpy())
        )

        layer = ToNumber(out_type=tf.float32, default_value=0.0)
        input = tf.constant([["123.1", ""], ["456", "-789.987"]], tf.string)
        output = layer.call(input)
        expected_output = tf.constant(
            [[123.1, 0.0], [456.0, -789.987]], tf.float32
        )
        self.assertEqual(output.dtype, tf.float32)
        self.assertTrue(
            np.array_equal(output.numpy(), expected_output.numpy())
        )

    def test_call_sparse(self):
        layer = ToNumber(out_type=tf.int32, default_value=-1)
        input = tf.SparseTensor(
            indices=[[0, 2], [2, 1], [2, 3], [5, 4]],
            values=tf.constant(["123", "", "456", "-789"], tf.string),
            dense_shape=[6, 5],
        )
        output = layer.call(input)
        expected_output = tf.SparseTensor(
            indices=[[0, 2], [2, 1], [2, 3], [5, 4]],
            values=tf.constant([123, -1, 456, -789], tf.int32),
            dense_shape=[6, 5],
        )
        self.assertTrue(sparse_tensor_equal(output, expected_output))

        layer = ToNumber(out_type=tf.float32, default_value=0.0)
        input = tf.SparseTensor(
            indices=[[0, 2], [2, 1], [2, 3], [5, 4]],
            values=tf.constant(["123.1", "", "456", "-789.987"], tf.string),
            dense_shape=[6, 5],
        )
        output = layer.call(input)
        expected_output = tf.SparseTensor(
            indices=[[0, 2], [2, 1], [2, 3], [5, 4]],
            values=tf.constant([123.1, 0.0, 456.0, -789.987], tf.float32),
            dense_shape=[6, 5],
        )
        self.assertTrue(sparse_tensor_equal(output, expected_output))


if __name__ == "__main__":
    unittest.main()
