import numpy as np
import tensorflow as tf

from elasticdl.proto import elasticdl_pb2, tensor_dtype_pb2
from elasticdl.python.common.dtypes import (
    dtype_numpy_to_tensor,
    dtype_tensor_to_numpy,
    is_numpy_dtype_allowed,
)


def tensor_to_ndarray(tensor_pb):
    """
    Create an ndarray from Tensor proto message. Note: upon return, the input
    tensor message is reset and underlying buffer passed to the returned
    ndarray.
    """

    if not tensor_pb.dim:
        raise ValueError("Tensor PB has no dim defined")

    # TODO (yunjian.lmh): Tensor of checkpoint file in `tests/testdata` does
    # not have `dtype` field. Remove default value for `tensor_pb.dtype` after
    # generating a new checkpoint file.
    if not tensor_pb.dtype:
        tensor_pb.dtype = tensor_dtype_pb2.DT_FLOAT32
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
    if not tensor_pb.indices:
        arr = np.ndarray(
            shape=tensor_pb.dim, dtype=dtype, buffer=tensor_pb.content
        )
    else:
        values = np.ndarray(
            shape=tensor_pb.dim, dtype=dtype, buffer=tensor_pb.content
        )
        indices = tf.convert_to_tensor(tensor_pb.indices)
        arr = tf.IndexedSlices(values, indices)
    tensor_pb.Clear()

    return arr


def ndarray_to_tensor(arr, indices=None):
    """Convert ndarray to Tensor PB"""

    if not is_numpy_dtype_allowed(arr.dtype):
        raise ValueError("Dtype of ndarray %s is not supported", arr.dtype)
    tensor = elasticdl_pb2.Tensor()
    tensor.dim.extend(arr.shape)
    tensor.content = arr.tobytes()
    if indices:
        tensor.indices.extend(indices)
    tensor.dtype = dtype_numpy_to_tensor(arr.dtype)

    return tensor
