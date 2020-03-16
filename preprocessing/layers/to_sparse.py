import tensorflow as tf
from tensorflow.python.feature_column.feature_column_v2 import (
    _to_sparse_input_and_drop_ignore_values,
)


class ToSparse(tf.keras.layers.Layer):
    def __init__(self):
        super(ToSparse, self).__init__()

    def call(self, inputs):
        return _to_sparse_input_and_drop_ignore_values(inputs)
