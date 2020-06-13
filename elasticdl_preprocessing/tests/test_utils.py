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
