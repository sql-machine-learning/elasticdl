import numpy as np
import tensorflow as tf
from tensorflow.core.framework import tensor_pb2

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.dtypes import (
    dtype_numpy_to_tensor,
    dtype_tensor_to_numpy,
)


def merge_indexed_slices(*args):
    return tf.IndexedSlices(
        tf.concat([i.values for i in args], axis=0),
        tf.concat([i.indices for i in args], axis=0),
    )


def deduplicate_indexed_slices(values, indices):
    """
    Sum up the values associated with duplicated indices and
    return unique indices with corresponding summed values.
    Args:
        values: A Tensor with rank >= 1.
        indices: A one-dimension integer of Tensor.
    Returns:
        A tuple of (`sum_combined_values`, `unique_indices`).
        `sum_combined_values` contains the sum of `values` associated
        with each unique indice.
        `unique_indices` is a de-duplicated version of `indices`.
    """
    unique_indices, new_index_positions = tf.unique(indices)
    sum_combined_values = tf.math.unsorted_segment_sum(
        values, new_index_positions, tf.shape(unique_indices)[0]
    )

    return (sum_combined_values, unique_indices)


def serialize_ndarray(array, pb):
    dtype = dtype_numpy_to_tensor(array.dtype)
    if not dtype:
        raise ValueError("Dtype of ndarray %s is not supported", array.dtype)
    pb.dtype = dtype
    pb.tensor_content = array.tobytes()
    for d in array.shape:
        pb_d = pb.tensor_shape.dim.add()
        pb_d.size = d


def ndarray_to_pb(array):
    pb = tensor_pb2.TensorProto()
    serialize_ndarray(array, pb)
    return pb


def pb_to_ndarray(pb):
    if not pb.tensor_shape:
        raise ValueError("PB has no dim defined")
    dtype = dtype_tensor_to_numpy(pb.dtype)
    size = dtype.itemsize
    shape = [d.size for d in pb.tensor_shape.dim]
    for d in shape:
        size *= d
    if size != len(pb.tensor_content):
        raise ValueError(
            "PB size mismatch, dim: %s, len(content): %d",
            str(shape),
            len(pb.tensor_content),
        )
    array = np.ndarray(shape=shape, dtype=dtype, buffer=pb.tensor_content)
    return array


def pb_to_indexed_slices(pb):
    concat_tensors = pb_to_ndarray(pb.concat_tensors)
    ids = np.array([int(i) for i in pb.ids])
    return tf.IndexedSlices(concat_tensors, ids)


def indexed_slices_to_pb(slices):
    pb = elasticdl_pb2.IndexedSlices()
    serialize_ndarray(slices.values, pb.concat_tensors)
    if (
        isinstance(slices.indices, np.ndarray)
        and len(slices.indices.shape) > 1
    ):
        raise ValueError(
            "IndexedSlices pb only accepts indices with one dimension, got %d",
            len(slices.indices.shape),
        )
    pb.ids.extend(slices.indices)
    return pb
