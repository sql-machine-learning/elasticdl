from __future__ import absolute_import, division, print_function

import tensorflow as tf


class Hashing(tf.keras.layers.Layer):
    """Distribute categorical feature values into a finite number of buckets
    by hashing.

    This layer converts a sequence of int or string to a sequence of int.
    output_id = Hash(input_feature_string) % num_bins for string type input.
    For int type input, the layer converts the value to string and then
    processes it by the same formula. TensorFlow 2.2 has developed
    `tf.keras.layers.preprocessing.Hashing` but not released it yet. So the
    layer is a simple temporary version.
    https://github.com/tensorflow/tensorflow/blob/r2.2/tensorflow/python/keras/layers/preprocessing/hashing.py

    Example:
    ```python
    layer = Hashing(num_bins=3)
    inp = np.asarray([['A'], ['B'], ['C'], ['D'], ['E']])
    layer(inp)
    ```
    The output will be `[[1], [0], [1], [1], [2]]`

    Arguments:
        num_bins: Number of hash bins.
        **kwargs: Keyword arguments to construct a layer.

    Input: A string, int32 or int64 `tf.Tensor`,
        `tf.SparseTensor` or `tf.RaggedTensor`

    Output shape: An int64 tensor with the same type as input.

    """

    def __init__(self, num_bins, **kwargs):
        if num_bins is None or num_bins <= 0:
            raise ValueError(
                "`num_bins` cannot be `None` or non-positive values."
            )
        super(Hashing, self).__init__(**kwargs)
        self.num_bins = num_bins
        self._supports_ragged_inputs = True

    def call(self, inputs):
        # Converts integer inputs to string.
        if inputs.dtype.is_integer:
            if isinstance(inputs, tf.SparseTensor):
                inputs = tf.SparseTensor(
                    indices=inputs.indices,
                    values=tf.as_string(inputs.values),
                    dense_shape=inputs.dense_shape,
                )
            else:
                inputs = tf.as_string(inputs)
        if isinstance(inputs, tf.RaggedTensor):
            return tf.ragged.map_flat_values(
                tf.strings.to_hash_bucket_fast,
                inputs,
                num_buckets=self.num_bins,
                name="hash",
            )
        elif isinstance(inputs, tf.SparseTensor):
            sparse_values = inputs.values
            sparse_hashed_values = tf.strings.to_hash_bucket_fast(
                sparse_values, self.num_bins, name="hash"
            )
            return tf.SparseTensor(
                indices=inputs.indices,
                values=sparse_hashed_values,
                dense_shape=inputs.dense_shape,
            )
        else:
            return tf.strings.to_hash_bucket_fast(
                inputs, self.num_bins, name="hash"
            )

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"num_bins": self.num_bins}
        base_config = super(Hashing, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
