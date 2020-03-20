import unittest

import tensorflow as tf

from elasticdl_preprocessing.layers.to_ragged import ToRagged
from elasticdl_preprocessing.tests.test_utils import ragged_tensor_equal


class ToRaggedTest(unittest.TestCase):
    def test_dense_to_ragged(self):
        layer = ToRagged()
        inp = tf.constant([[1], [-1], [4]], tf.int64)
        out = layer(inp)
        expected_out = tf.ragged.constant([[1], [], [4]], tf.int64)
        self.assertTrue(ragged_tensor_equal(out, expected_out))

    def test_string_split_to_ragged(self):
        layer = ToRagged()
        inp = tf.constant([["1,2,3"], ["4,5"], [""]])
        out = layer(inp)
        expected_out = tf.ragged.constant([["1", "2", "3"], ["4", "5"], []])
        self.assertTrue(ragged_tensor_equal(out, expected_out))


if __name__ == "__main__":
    unittest.main()
