# Copyright 2020 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from odps import types
from tensorflow.core.framework import types_pb2


def dtype_tensor_to_numpy(dtype):
    """Convert tensor dtype to numpy dtype object."""
    np_dtype_name = _DT_TENSOR_TO_NP.get(dtype, None)
    if not np_dtype_name:
        raise ValueError("Got wrong tensor PB dtype %s.", dtype)
    return np.dtype(np_dtype_name)


def dtype_numpy_to_tensor(dtype):
    """Convert numpy dtype object to tensor dtype."""
    return _DT_NP_TO_TENSOR.get(dtype.type, types_pb2.DT_INVALID)


def is_numpy_dtype_allowed(dtype):
    return dtype.type in _DT_NP_TO_TENSOR


_DT_TENSOR_TO_NP = {
    types_pb2.DT_INT8: np.int8,
    types_pb2.DT_INT16: np.int16,
    types_pb2.DT_INT32: np.int32,
    types_pb2.DT_INT64: np.int64,
    types_pb2.DT_FLOAT: np.float32,
    types_pb2.DT_DOUBLE: np.float64,
    types_pb2.DT_BOOL: np.bool,
}

_DT_NP_TO_TENSOR = {
    np.int8: types_pb2.DT_INT8,
    np.int16: types_pb2.DT_INT16,
    np.int32: types_pb2.DT_INT32,
    np.int64: types_pb2.DT_INT64,
    np.float32: types_pb2.DT_FLOAT,
    np.float64: types_pb2.DT_DOUBLE,
    np.bool: types_pb2.DT_BOOL,
}

# TODO: There are many dtypes in MaxCompute and we can add them if needed.
MAXCOMPUTE_DTYPE_TO_TF_DTYPE = {
    types.bigint: types_pb2.DT_INT64,
    types.double: types_pb2.DT_DOUBLE,
    types.string: types_pb2.DT_STRING,
}
