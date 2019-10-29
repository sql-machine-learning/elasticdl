import numpy as np
import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.dtypes import (
    dtype_numpy_to_tensor,
    dtype_tensor_to_numpy,
)
from elasticdl.python.common.log_utils import default_logger as logger


def serialize_tensor(tensor, tensor_pb):
    """Serialize ElasticDL Tensor to tensor protocol buffer."""
    dtype = dtype_numpy_to_tensor(tensor.values.dtype)
    if not dtype:
        raise ValueError(
            "Dtype of ndarray %s is not supported", tensor.values.dtype
        )
    tensor_pb.dtype = dtype
    tensor_pb.dim.extend(tensor.values.shape)
    tensor_pb.content = tensor.values.tobytes()
    if tensor.is_indexed_slices():
        tensor_pb.indices.extend(tuple(tensor.indices))
    if tensor.name:
        tensor_pb.name = tensor.name


def deserialize_tensor_pb(tensor_pb, tensor):
    """Deserialize tensor protocol buffer to ElasticDL Tensor.

    Note that the input tensor protocol buffer is reset and underlying buffer
    is passed to the returned ndarray.
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
    tensor.set(
        np.ndarray(shape=tensor_pb.dim, dtype=dtype, buffer=tensor_pb.content),
        np.array(tensor_pb.indices) if tensor_pb.indices else None,
        tensor_pb.name,
    )
    tensor_pb.Clear()


def tensor_pb_to_ndarray(tensor_pb):
    return Tensor.from_tensor_pb(tensor_pb).to_ndarray()


def tensor_pb_to_tf_tensor(tensor_pb):
    return Tensor.from_tensor_pb(tensor_pb).to_tf_tensor()


def append_tensor_pb_from_ndarray(
    tensor_pb_list, values, indices=None, name=None
):
    tensor_pb = tensor_pb_list.add()
    tensor = Tensor(values, indices, name)
    serialize_tensor(tensor, tensor_pb)


class Tensor(object):
    """Data structure for tensors in ElasticDL.

    `Tensor` can save dense tensors and sparse tensors. For sparse tensors,
    this structure saves them in the same way as `TensorFlow.IndexedSlices`.
    """

    def __init__(self, values=None, indices=None, name=None):
        """
        `Tensor` can save dense tensors and sparse tensors.
        To pass in a dense tensor, `values` should be `numpy.ndarray` and
            `indices` should be None.
        There are two ways to pass in a sparse tensor:
            * `values` is a `numpy.ndarray` and `indices` is a `numpy.ndarray`.
            * `values` is a `TensorFlow.IndexedSlices` and `indices` is None.

        Args:
            values: A `numpy.ndarray` or `TensorFlow.IndexedSlices`.
                If `values` is a `TensorFlow.IndexedSlices`, `indices` should
                be None.
            indices: A `numpy.ndarray` or None.
            name: A python string.
        """
        self.set(values, indices, name)

    @classmethod
    def from_tensor_pb(cls, tensor_pb):
        tensor = cls()
        deserialize_tensor_pb(tensor_pb, tensor)
        return tensor

    @classmethod
    def from_tf_variable(cls, tf_variable):
        return cls(tf_variable.numpy(), name=tf.variable.name)

    def set(self, values=None, indices=None, name=None):
        self.name = name
        if isinstance(values, tf.IndexedSlices):
            if indices is not None:
                raise ValueError(
                    "When creating a Tensor object with values of type "
                    "tf.IndexedSlices, indices must be None."
                )
            if values.dense_shape is not None:
                # TODO(yunjian.lmh): Support dense shape, or do not print
                #     warning message, or there will be too much warning
                #     warning messages.
                logger.warning(
                    "Convert a TensorFlow.IndexedSlices object with dense "
                    "shape to ElasticDL Tensor. ElasticDL ignores dense shape "
                    "now."
                )

            self.values = values.values.numpy()
            self.indices = values.indices.numpy()
        else:
            self.values = (
                values.numpy() if isinstance(values, tf.Tensor) else values
            )
            self.indices = (
                indices.numpy() if isinstance(indices, tf.Tensor) else indices
            )

    def is_indexed_slices(self):
        return self.indices is not None

    def to_tensor_pb(self):
        tensor_pb = elasticdl_pb2.Tensor()
        serialize_tensor(self, tensor_pb)
        return tensor_pb

    def to_tf_tensor(self):
        if self.is_indexed_slices():
            return tf.IndexedSlices(self.values, self.indices)
        else:
            return tf.constant(self.values)

    def to_ndarray(self):
        if self.is_indexed_slices():
            # Currently Tensor does not have a field representing dense shape,
            # thus can not convert it to numpy.ndarray.
            raise NotImplementedError(
                "Converting an ElasticDL Tensor object, which contains a "
                "sparse tensor, to a numpy.ndarray is not supported."
            )
        return np.array(self.values)
