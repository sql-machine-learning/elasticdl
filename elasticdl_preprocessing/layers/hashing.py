from __future__ import absolute_import, division, print_function

from tensorflow.python.framework import dtypes, sparse_tensor, tensor_spec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_functional_ops, ragged_tensor


class Hashing(Layer):
    """Implements categorical feature hashing, also known as "hashing trick".

    This layer transforms categorical inputs to hashed output. It converts a
    sequence of int or string to a sequence of int.

    Example:
    ```python
    layer = Hashing(num_bins=3)
    inp = np.asarray([['A'], ['B'], ['C'], ['D'], ['E']])
    layer(inputs)
    [[1], [0], [1], [1], [2]]
    ```

    Arguments:
    num_bins: Number of hash bins.
    name: Name to give to the layer.
    **kwargs: Keyword arguments to construct a layer.

    Input shape: A string, int32 or int64 tensor of shape
    `[batch_size, d1, ..., dm]`

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
            if isinstance(inputs, sparse_tensor.SparseTensor):
                inputs = sparse_tensor.SparseTensor(
                    indices=inputs.indices,
                    values=string_ops.as_string(inputs.values),
                    dense_shape=inputs.dense_shape,
                )
            else:
                inputs = string_ops.as_string(inputs)
        if ragged_tensor.is_ragged(inputs):
            return ragged_functional_ops.map_flat_values(
                string_ops.string_to_hash_bucket_fast,
                inputs,
                num_buckets=self.num_bins,
                name="hash",
            )
        elif isinstance(inputs, sparse_tensor.SparseTensor):
            sparse_values = inputs.values
            sparse_hashed_values = string_ops.string_to_hash_bucket_fast(
                sparse_values, self.num_bins, name="hash"
            )
            return sparse_tensor.SparseTensor(
                indices=inputs.indices,
                values=sparse_hashed_values,
                dense_shape=inputs.dense_shape,
            )
        else:
            return string_ops.string_to_hash_bucket_fast(inputs, self.num_bins, name="hash")

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_output_signature(self, input_spec):
        output_shape = self.compute_output_shape(input_spec.shape.as_list())
        output_dtype = dtypes.int64
        if isinstance(input_spec, sparse_tensor.SparseTensorSpec):
            return sparse_tensor.SparseTensorSpec(
                shape=output_shape, dtype=output_dtype
            )
        else:
            return tensor_spec.TensorSpec(
                shape=output_shape, dtype=output_dtype
            )

    def get_config(self):
        config = {"num_bins": self.num_bins}
        base_config = super(Hashing, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
