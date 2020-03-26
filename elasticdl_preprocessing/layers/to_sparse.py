import tensorflow as tf


class ToSparse(tf.keras.layers.Layer):
    """Converts a `Tensor` to a `SparseTensor`, dropping ignore_value cells.
    If the input is already a `SparseTensor`, just return it.

    Example :
    ```python
        layer = ToSparse()
        inp = tf.constant([["A", ""], ["B", "C"]], tf.string)
        out = layer(inp)
    ```
    The expected output is
    ```
    tf.SparseTensor(
            indices=np.array([[0, 0], [1, 0], [1, 1]]),
            values=np.array(["A", "B", "C"]),
            dense_shape=(2, 2),
        )
    ```

    Arguments:
        ignore_value: Entries in inputs equal to this value will be
            absent from the output `SparseTensor`. If `None`, default value of
            inputs dtype will be used ('' for `str`, -1 for `int`).

    Input shape: A numeric or string `Tensor` of shape
        `[batch_size, d1, ..., dm]`

    Output shape: An `SparseTensor` with the same shape as inputs
    """

    def __init__(self, ignore_value=None):
        super(ToSparse, self).__init__()
        self.ignore_value = ignore_value

    def call(self, inputs):
        if isinstance(inputs, tf.SparseTensor):
            return inputs

        ignore_value = self.ignore_value
        if ignore_value is None:
            if inputs.dtype == tf.string:
                ignore_value = ""
            elif inputs.dtype.is_integer:
                ignore_value = -1
            else:
                ignore_value = 0.0
        ignore_value = tf.cast(ignore_value, inputs.dtype)
        indices = tf.where(tf.not_equal(inputs, ignore_value))
        values = tf.gather_nd(inputs, indices)
        dense_shape = tf.shape(inputs, out_type=tf.int64)
        return tf.SparseTensor(
            indices=indices, values=values, dense_shape=dense_shape
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "ignore_value": self.ignore_value,
        }
        base_config = super(ToSparse, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
