import tensorflow as tf

_COMMA_SEP = ","


class ToRagged(tf.keras.layers.Layer):
    """Converts a `Tensor` or `RaggedTensor` to a `RaggedTensor`,
    dropping ignore_value cells. If the input's dtype is string, split
    the string elements to convert the input to `RaggedTensor` firstly.

    Example (Integer):
    ```python
        layer = ToRagged()
        inp = tf.constant([[1], [-1], [4]], tf.int64)
        out = layer(inp)
        [[1], [], [4]]
    ```

    Example (String):
    ```python
        ayer = ToRagged()
        inp = tf.constant([["1,2,3"], ["4,5"], [""]])
        out = layer(inp)
    ```
    The expected output is `[["1", "2", "3"], ["4", "5"], []]`

    Arguments:
        sep: Valid if the input's dtype is string.
        ignore_value: Entries in inputs equal to this value will be
            absent from the output `RaggedTensor`. If `None`, default value of
            input's dtype will be used ('' for `str`, -1 for `int`).

    Input shape: A numeric or string `Tensor` or `RaggedTensor` of shape
        `[batch_size, d1, ..., dm]`

    Output shape: An `RaggedTensor` with the same shape as inputs
    """

    def __init__(self, sep=_COMMA_SEP, ignore_value=None):
        super(ToRagged, self).__init__()
        self.sep = sep
        self.ignore_value = ignore_value

    def call(self, inputs):
        if (
            not isinstance(inputs, tf.Tensor) or
            not isinstance(inputs, tf.RaggedTensor)
        ):
            raise TypeError(
                "The inputs must be a Tensor or RaggedTensor and "
                "the type of inputs is {}".format(type(inputs))
            )

        if isinstance(inputs, tf.Tensor):
            inputs = tf.RaggedTensor.from_tensor(inputs)

        if self.ignore_value is None:
            if inputs.dtype == tf.string:
                self.ignore_value = ""
                inputs = tf.strings.split(inputs, sep=self.sep).values
            elif inputs.dtype.is_integer:
                self.ignore_value = -1

        if self.ignore_value is not None:
            self.ignore_value = tf.cast(self.ignore_value, inputs.dtype)
            return tf.ragged.boolean_mask(
                inputs, tf.not_equal(inputs, self.ignore_value)
            )

        return inputs
