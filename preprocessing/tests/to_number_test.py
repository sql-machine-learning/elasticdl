import tensorflow as tf 
import unittest
from preprocessing.layers.to_number import ToNumber

class ToNumberTest(unittest.TestCase):
    def test_default(self):
        to_number_layer = ToNumber(out_type=tf.int32, default_value=-1)
        self.assertIsNotNone(to_number_layer)
