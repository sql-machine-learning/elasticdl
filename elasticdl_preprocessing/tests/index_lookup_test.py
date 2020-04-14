import os
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from elasticdl_preprocessing.layers.index_lookup import IndexLookup
from elasticdl_preprocessing.tests.test_utils import (
    ragged_tensor_equal,
    sparse_tensor_equal,
)


class IndexLookupTest(unittest.TestCase):
    def test_lookup_with_list(self):
        lookup_layer = IndexLookup(vocabulary=["A", "B", "C"])
        self._check_lookup(lookup_layer)
        self.assertEqual(lookup_layer.vocab_size(), 4)

    def test_lookup_with_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vocab_file = os.path.join(temp_dir, "vocab_test.txt")
            with open(vocab_file, "w") as f:
                f.write("A\n")
                f.write("B\n")
                f.write("C\n")
            lookup_layer = IndexLookup(vocabulary=vocab_file)
            self._check_lookup(lookup_layer)

    def test_model_with_lookup(self):
        inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
        lookup_out = IndexLookup(vocabulary=["A", "B", "C"])(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=lookup_out)
        out = model.call(tf.constant([["A"], ["C"], ["B"], ["D"], ["E"]]))
        self.assertTrue(
            np.array_equal(
                np.array([[0], [2], [1], [3], [3]], dtype=int), out.numpy()
            )
        )

    def _check_lookup(self, lookup_layer):
        dense_input = tf.constant([["A"], ["B"], ["C"], ["D"], ["E"]])
        output = lookup_layer(dense_input)
        expected_out = np.array([[0], [1], [2], [3], [3]])
        self.assertTrue(np.array_equal(output.numpy(), expected_out))

        ragged_input = tf.ragged.constant([["A", "B", "C"], ["D", "E"]])
        ragged_output = lookup_layer(ragged_input)
        expected_ragged_out = tf.ragged.constant(
            [[0, 1, 2], [3, 3]], dtype=tf.int64
        )
        self.assertTrue(
            ragged_tensor_equal(ragged_output, expected_ragged_out)
        )

        sparse_input = ragged_input.to_sparse()
        sparse_output = lookup_layer(sparse_input)
        expected_sparse_out = expected_ragged_out.to_sparse()
        self.assertTrue(
            sparse_tensor_equal(sparse_output, expected_sparse_out)
        )
