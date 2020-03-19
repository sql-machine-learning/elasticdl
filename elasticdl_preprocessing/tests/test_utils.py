import numpy as np


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
