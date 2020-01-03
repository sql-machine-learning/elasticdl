import numpy as np

from elasticdl.proto import elasticdl_pb2


def dtype_tensor_to_numpy(dtype):
    """Convert tensor dtype to numpy dtype object."""
    np_dtype_name = _DT_TENSOR_TO_NP.get(dtype, None)
    if not np_dtype_name:
        raise ValueError("Got wrong tensor PB dtype %s.", dtype)
    return np.dtype(np_dtype_name)


def dtype_numpy_to_tensor(dtype):
    """Convert numpy dtype object to tensor dtype."""
    return _DT_NP_TO_TENSOR.get(dtype.type, elasticdl_pb2.DT_INVALID)


def is_numpy_dtype_allowed(dtype):
    return dtype.type in _DT_NP_TO_TENSOR


_DT_TENSOR_TO_NP = {
    elasticdl_pb2.DT_INT8: np.int8,
    elasticdl_pb2.DT_INT16: np.int16,
    elasticdl_pb2.DT_INT32: np.int32,
    elasticdl_pb2.DT_INT64: np.int64,
    elasticdl_pb2.DT_FLOAT16: np.float16,
    elasticdl_pb2.DT_FLOAT32: np.float32,
    elasticdl_pb2.DT_FLOAT64: np.float64,
    elasticdl_pb2.DT_BOOL: np.bool,
}

_DT_NP_TO_TENSOR = {
    np.int8: elasticdl_pb2.DT_INT8,
    np.int16: elasticdl_pb2.DT_INT16,
    np.int32: elasticdl_pb2.DT_INT32,
    np.int64: elasticdl_pb2.DT_INT64,
    np.float16: elasticdl_pb2.DT_FLOAT16,
    np.float32: elasticdl_pb2.DT_FLOAT32,
    np.float64: elasticdl_pb2.DT_FLOAT64,
    np.bool: elasticdl_pb2.DT_BOOL,
}
