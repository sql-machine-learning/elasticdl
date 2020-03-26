import numpy as np
import tensorflow as tf


def sparse_tensor_equal(sp_a, sp_b):
    if not isinstance(sp_a, tf.SparseTensor) or not isinstance(
        sp_b, tf.SparseTensor
    ):
        return False

    if not np.array_equal(sp_a.dense_shape.numpy(), sp_b.dense_shape.numpy()):
        return False

    if not np.array_equal(sp_a.indices.numpy(), sp_b.indices.numpy()):
        return False

    if sp_a.values.dtype != sp_b.values.dtype:
        return False

    if not np.array_equal(sp_a.values.numpy(), sp_b.values.numpy()):
        return False

    return True


def ragged_tensor_equal(rt_a, rt_b):
    if not isinstance(rt_a, tf.RaggedTensor) or not isinstance(
        rt_b, tf.RaggedTensor
    ):
        return False

    if rt_a.shape.as_list() != rt_b.shape.as_list():
        return False

    if rt_a.dtype != rt_b.dtype:
        return False

    if rt_a.to_list() != rt_b.to_list():
        return False

    return True
