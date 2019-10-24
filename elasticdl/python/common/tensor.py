import numpy as np
import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.dtypes import (
    dtype_numpy_to_tensor,
    dtype_tensor_to_numpy,
    is_numpy_dtype_allowed,
)


def serialize_tensor(tensor, tensor_pb):
    """Convert Tensor to Tensor PB"""
    dtype = tensor.values.dtype
    if not is_numpy_dtype_allowed(dtype):
        raise ValueError("Dtype of ndarray %s is not supported", dtype)
    tensor_pb.dim.extend(tensor.values.shape)
    tensor_pb.content = tensor.values.tobytes()
    if tensor.is_indexed_slices():
        tensor_pb.indices.extend(tensor.indices)
    tensor_pb.dtype = dtype_numpy_to_tensor(dtype)
    if tensor.name:
        tensor_pb.name = tensor.name


def deserialize_tensor_pb(tensor_pb, tensor):
    """Create an Tensor instance from Tensor proto message.

    Note: upon return, the input tensor message is reset and underlying buffer
    passed to the returned ndarray.
    """

    if not tensor_pb.dim:
        raise ValueError("Tensor PB has no dim defined")

    dtype = dtype_tensor_to_numpy(tensor_pb.dtype)
    # Check that the buffer size agrees with dimensions.
    size = dtype.itemsize
    for d in tensor_pb.dim:
        size *= d
    if size != len(tensor_pb.content):
        raise ValueError(
            "Tensor PB size mismatch, dim: %s, len(content): %d",
            tensor_pb.dim,
            len(tensor_pb.content),
        )
    tensor = Tensor()
    tensor.set(
        np.ndarray(shape=tensor_pb.dim, dtype=dtype, buffer=tensor_pb.content),
        tensor_pb.indices,
        tensor_pb.name,
    )
    tensor_pb.Clear()


class Tensor(object):
    """Data structure for tensors in ElasticDL.

    `Tensor` can save dense tensors and sparse tensors. For sparse tensors,
    this structure saves them in the same way as `TensorFlow.IndexedSlices`.
    """

    def __init__(self, values=None, indices=None, name=None):
        """
        `Tensor` can save dense tensors and sparse tensors.
        To save dense tensors, `values` should be `numpy.ndarray` and `indices`
            should be None.
        There are two ways to save sparse tensors:
            * `values` is a `numpy.ndarray` and `indices` is a `numpy.ndarray`.
            * `values` is a `TensorFlow.IndexedSlices` and `indices` is None.

        Args:
            values: A `numpy.ndarray` or `TensorFlow.IndexedSlices`.
                If `values` is a `TensorFlow.IndexedSlices`, `indices` should
                be None.
            indices: `indices` of `TensorFlow`
        """
        self.set(values, indices, name)

    def set(self, values=None, indices=None, name=None):
        self.name = name
        if isinstance(values, tf.IndexedSlices):
            if indices is not None:
                raise ValueError(
                    "When creating a Tensor instance with values of type "
                    "tf.IndexedSlices, indices should be None."
                )
            self.values = values.values.numpy()
            self.indices = values.indices.numpy()
        else:
            self.values = values
            self.indices = indices

    def is_indexed_slices(self):
        return self.indices is not None

    def from_tensor_pb(self, tensor_pb):
        """Parse tensor from Tensor proto message.

        Note: upon return, the input tensor message is reset and underlying
        buffer passed to the returned ndarray.
        """
        deserialize_tensor_pb(tensor_pb, self)

    def to_tensor_pb(self):
        tensor_pb = elasticdl_pb2.Tensor()
        serialize_tensor(self, tensor_pb)
        return tensor_pb

    def to_tf_tensor(self):
        if self.is_indexed_slices():
            return tf.IndexedSlices(self.values, self.indices)
        else:
            return tf.constant(self.values)
