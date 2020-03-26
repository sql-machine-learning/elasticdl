import unittest

import numpy as np
import tensorflow as tf

from elasticdl_preprocessing.layers.to_sparse import ToSparse
from elasticdl_preprocessing.tests.test_utils import sparse_tensor_equal


class ToSparseTest(unittest.TestCase):
    def test_to_sparse(self):
        layer = ToSparse()
        inp = tf.constant([["A", ""], ["B", "C"]], tf.string)
        output = layer.call(inp)
        expected_out = tf.SparseTensor(
            indices=np.array([[0, 0], [1, 0], [1, 1]]),
            values=np.array(["A", "B", "C"]),
            dense_shape=(2, 2),
        )
        self.assertTrue(sparse_tensor_equal(output, expected_out))

        layer = ToSparse()
        inp = tf.constant([[12, -1], [45, 78]], tf.int64)
        output = layer.call(inp)
        expected_out = tf.SparseTensor(
            indices=np.array([[0, 0], [1, 0], [1, 1]]),
            values=np.array([12, 45, 78]),
            dense_shape=(2, 2),
        )
        self.assertTrue(sparse_tensor_equal(output, expected_out))

    def test_model_with_to_sparse(self):
        inputs = tf.keras.Input(shape=(1,), dtype=tf.int32)
        sparse_inputs = ToSparse(ignore_value=-1)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=sparse_inputs)
        out = model.call(tf.constant([[1], [-1], [2], [3]]))

        expect_out = tf.SparseTensor(
            indices=tf.constant([[0, 0], [2, 0], [3, 0]], dtype=tf.int64),
            values=tf.constant([1, 2, 3], dtype=tf.int32),
            dense_shape=(4, 1),
        )
        self.assertTrue(sparse_tensor_equal(out, expect_out))


if __name__ == "__main__":
    unittest.main()
