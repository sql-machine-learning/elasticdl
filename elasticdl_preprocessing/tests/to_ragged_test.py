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

    def test_model_with_ragged(self):
        inputs = tf.keras.Input(shape=(1,), dtype=tf.int32)
        ragged = ToRagged(ignore_value=-1)(inputs)
        sum_out = tf.reduce_sum(ragged)
        model = tf.keras.Model(inputs=inputs, outputs=sum_out)
        sum_value = model.call(tf.constant([[1], [-1], [4]])).numpy()
        self.assertEqual(sum_value, 5.0)


if __name__ == "__main__":
    unittest.main()
