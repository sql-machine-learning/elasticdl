import os

import tensorflow as tf
from tensorflow.python.ops import lookup_ops, math_ops

"""
The layers in this file will be replaced with the layers in
elasticdl_preprocessing folder while they are committed.
"""

class ToSparse(tf.keras.layers.Layer):
    """Converts a `Tensor` to a `SparseTensor`, dropping ignore_value cells.
    If the input is already a `SparseTensor`, just return it.
    Example :
    ```python
        layer = ToSparse()
        inp = tf.constant([["A", ""], ["B", "C"]], tf.string)
        layer.call(inp)
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
        if self.ignore_value is None:
            if inputs.dtype == tf.string:
                ignore_value = ""
            elif inputs.dtype.is_integer:
                ignore_value = -1
            else:
                ignore_value = -1
        else:
            ignore_value = self.ignore_value

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


class Discretization(tf.keras.layers.Layer):
    """Buckets data into discrete ranges.
    TensorFlow 2.2 has developed `tf.keras.layers.preprocessing.Discretization`
    but not released it yet. So the layer is a simple temporary version
    `tensorflow.python.keras.layers.preprocessing.discretization.Discretization`
    Input shape:
        Any `tf.Tensor` or `tf.RaggedTensor` of dimension 2 or higher.
    Output shape:
        The same as the input shape with tf.int64.
    Attributes:
        bins: Optional boundary specification. Bins include the left boundary
            and exclude the right boundary, so `bins=[0., 1., 2.]` generates
            bins `(-inf, 0.)`, `[0., 1.)`, `[1., 2.)`, and `[2., +inf)`.
  """

    def __init__(self, bins, **kwargs):
        super(Discretization, self).__init__(**kwargs)
        self._supports_ragged_inputs = True
        self.bins = bins

    def get_config(self):
        config = {
            "bins": self.bins,
        }
        base_config = super(Discretization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        if isinstance(inputs, tf.RaggedTensor):
            integer_buckets = tf.ragged.map_flat_values(
                math_ops._bucketize, inputs, boundaries=self.bins
            )
            integer_buckets = tf.identity(integer_buckets)
        elif isinstance(inputs, tf.SparseTensor):
            integer_bucket_values = math_ops._bucketize(
                inputs.values, boundaries=self.bins
            )
            integer_buckets = tf.SparseTensor(
                indices=inputs.indices,
                values=integer_bucket_values,
                dense_shape=inputs.dense_shape,
            )
        else:
            integer_buckets = math_ops._bucketize(inputs, boundaries=self.bins)

        return tf.cast(integer_buckets, tf.int64)


class Hashing(tf.keras.layers.Layer):
    """Implements categorical feature hashing, also known as "hashing trick".
    This layer transforms categorical inputs to hashed output. It converts a
    sequence of int or string to a sequence of int. TensorFlow 2.2 has
    developed `tf.keras.layers.preprocessing.Hashing` but not released it yet.
    So the layer is a simple temporary version.
    https://github.com/tensorflow/tensorflow/blob/r2.2/tensorflow/python/keras/layers/preprocessing/hashing.py
    Example:
    ```python
    layer = Hashing(num_bins=3)
    inp = np.asarray([['A'], ['B'], ['C'], ['D'], ['E']])
    layer(inp)
    [[1], [0], [1], [1], [2]]
    ```
    Arguments:
        num_bins: Number of hash bins.
        name: Name to give to the layer.
        **kwargs: Keyword arguments to construct a layer.
    Input shape: A string, int32 or int64 tensor of shape
        `[batch_size, d1, ..., dm]`. The tensor can be `tf.Tensor`,
        `tf.SparseTensor` and `tf.RaggedTensor`
    Output shape: An int64 tensor of shape `[batch_size, d1, ..., dm]`
    """

    def __init__(self, num_bins, name=None, **kwargs):
        if num_bins is None or num_bins <= 0:
            raise ValueError(
                "`num_bins` cannot be `None` or non-positive values."
            )
        super(Hashing, self).__init__(name=name, **kwargs)
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


class Lookup(tf.keras.layers.Layer):
    """
    Todo replace it with a preprocess layer after TF 2.2
    https://github.com/tensorflow/community/pull/188/files?short_path=0657914#diff-0657914a8dc40e5fbca67680bf3fc45f
    """

    def __init__(
        self,
        vocabulary_list=None,
        vocabulary_file=None,
        num_oov_buckets=0,
        default_value=-1,
        vocabulary_size=None,
    ):
        super(Lookup, self).__init__()
        self.vocabulary_list = vocabulary_list
        self.num_oov_buckets = num_oov_buckets
        self.default_value = default_value
        self.vocabulary_size = vocabulary_size

        if vocabulary_file is not None:
            self.vocabulary_list = self._get_vocab_list_from_file(
                vocabulary_file
            )

    def build(self, input_shape):
        self.table = lookup_ops.index_table_from_tensor(
            vocabulary_list=self.vocabulary_list,
            num_oov_buckets=self.num_oov_buckets,
            default_value=self.default_value,
        )

    def call(self, inputs):
        if isinstance(inputs, tf.SparseTensor):
            lookup_id = self.table.lookup(inputs.values)
            output = tf.SparseTensor(
                indices=inputs.indices,
                values=lookup_id,
                dense_shape=inputs.dense_shape,
            )
        else:
            output = self.table.lookup(inputs)
        return tf.cast(output, tf.int64)

    def _get_vocab_list_from_file(self, file_path):
        vocab = []
        if os.path.exists(file_path):
            with open(file_path) as f:
                for line in f.readlines():
                    vocab.append(line.strip())
        return vocab


class ConcatenateWithOffset(tf.keras.layers.Concatenate):
    """Layer that add offset for tensor in the list of inputs and
    concatenate the tensors.
    It takes as input a list of tensors and returns a single tensor.
    Firstly, it will add an offset in offsets for each tensor in inputs.
    Then concatenate them to a single tensor. The tensor in inputs
    must have the same type, `Tensor` or `RaggedTensor` or `SparseTensor` and
    the same shape.
    Example :
    ```python
        a1 = tf.constant([[1], [1], [1]])
        a2 = tf.constant([[2], [2], [2]])
        offsets = [0, 10]
        layer = ConcatenateWithOffset(offsets=offsets, axis=1)
        layer([a1, a2])
        [[ 1 12]
         [ 1 12]
         [ 1 12]]
    ```
    Arguments:
        offsets: numeric list to add
        axis: Axis along which to concatenate.
        **kwargs: standard layer keyword arguments.
    """

    def __init__(self, offsets, axis=-1):
        super(ConcatenateWithOffset, self).__init__()
        self.offsets = offsets
        self.axis = axis

    def call(self, inputs):
        ids_with_offset = []
        if len(self.offsets) != len(inputs):
            raise ValueError(
                "The offsets length is not equal to inputs length"
                "the inputs are {}, offsets are {}".format(
                    inputs, self.offsets
                )
            )
        for i, tensor in enumerate(inputs):
            if isinstance(tensor, tf.SparseTensor):
                ids_with_offset.append(
                    tf.SparseTensor(
                        indices=tensor.indices,
                        values=tensor.values + self.offsets[i],
                        dense_shape=tensor.dense_shape,
                    )
                )
            else:
                ids_with_offset.append(tensor + self.offsets[i])

        if isinstance(ids_with_offset[0], tf.SparseTensor):
            result = tf.sparse.concat(
                axis=self.axis, sp_inputs=ids_with_offset
            )
        else:
            result = tf.keras.layers.concatenate(
                ids_with_offset, axis=self.axis
            )
        return result
