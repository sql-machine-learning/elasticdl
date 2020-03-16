import tensorflow as tf


class ToNumber(tf.keras.layers.Layer):
    def __init__(self, out_type, default_value):
        super(ToNumber, self).__init__()
        assert out_type in [
            tf.int16,
            tf.int32,
            tf.int64,
            tf.float16,
            tf.float32,
            tf.float64,
            tf.double,
        ]
        self.out_type = out_type
        self.default_value = default_value

    def call(self, inputs):
        if isinstance(inputs, tf.SparseTensor):
            number_value = self._cast_dense_to_number(inputs.values)
            return tf.SparseTensor(
                indices=inputs.indices,
                values=number_value,
                dense_shape=inputs.dense_shape,
            )
        else:
            return self._cast_dense_to_number(inputs)

    def _cast_dense_to_number(self, dense_inputs):
        if dense_inputs.dtype is tf.string:
            default_value = str(self.default_value)
            outputs = tf.where(
                tf.equal(dense_inputs, ""), x=default_value, y=dense_inputs
            )
            outputs = tf.strings.to_number(outputs, out_type=self.out_type)
        else:
            outputs = tf.cast(dense_inputs, self.out_type)

        return outputs
