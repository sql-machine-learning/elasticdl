import numpy as np

from elasticdl.proto import tensor_dtype_pb2


def dtype_tensor_to_numpy(dtype):
    """Convert tensor dtype to numpy dtype object."""
    return np.dtype(_DT_TENSOR_TO_NP.get(dtype, None))


def dtype_numpy_to_tensor(dtype):
    """Convert numpy dtype object to tensor dtype."""
    return _DT_NP_TO_TENSOR.get(dtype.type, tensor_dtype_pb2.DT_INVALID)


_DT_TENSOR_TO_NP = {
    tensor_dtype_pb2.DT_INT8: np.int8,
    tensor_dtype_pb2.DT_INT16: np.int16,
    tensor_dtype_pb2.DT_INT32: np.int32,
    tensor_dtype_pb2.DT_INT64: np.int64,
    tensor_dtype_pb2.DT_FLOAT16: np.float16,
    tensor_dtype_pb2.DT_FLOAT32: np.float32,
    tensor_dtype_pb2.DT_FLOAT64: np.float64,
    tensor_dtype_pb2.DT_BOOL: np.bool,
}

_DT_NP_TO_TENSOR = {
    np.int8: tensor_dtype_pb2.DT_INT8,
    np.int16: tensor_dtype_pb2.DT_INT16,
    np.int32: tensor_dtype_pb2.DT_INT32,
    np.int64: tensor_dtype_pb2.DT_INT64,
    np.float16: tensor_dtype_pb2.DT_FLOAT16,
    np.float32: tensor_dtype_pb2.DT_FLOAT32,
    np.float64: tensor_dtype_pb2.DT_FLOAT64,
    np.bool: tensor_dtype_pb2.DT_BOOL,
}
