import numpy as np
import tensorflow as tf


def sparse_tensor_equal(sp_a, sp_b):
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

    for i in range(rt_a.shape[0]):
        sub_rt_a = rt_a[i]
        sub_rt_b = rt_b[i]
        if isinstance(sub_rt_a, tf.RaggedTensor) and isinstance(
            sub_rt_b, tf.RaggedTensor
        ):
            if not ragged_tensor_equal(sub_rt_a, sub_rt_b):
                return False
        elif isinstance(sub_rt_a, tf.Tensor) and isinstance(
            sub_rt_b, tf.Tensor
        ):
            if sub_rt_a.dtype != sub_rt_b.dtype:
                return False
            if not np.array_equal(sub_rt_a.numpy(), sub_rt_b.numpy()):
                return False
        else:
            return False
    return True
