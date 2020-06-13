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

import tensorflow as tf
from tensorflow.python.ops.ragged import ragged_functional_ops, ragged_tensor


class RoundIdentity(tf.keras.layers.Layer):
    """Cast a numeric feature into a discrete integer value.

    This layer transforms numeric inputs to integer output. It is a special
    case of bucketizing to bins. The max value in the layer is the number of
    bins.

    Example :
    ```python
        layer = RoundIdentity(max_value=5)
        inp = np.asarray([[1.2], [1.6], [0.2], [3.1], [4.9]])
        layer(inp)
        [[1], [2], [0], [3], [5]]
    ```

    Arguments:
        num_buckets: Range of inputs and outputs is `[0, num_buckets)`.
        **kwargs: Keyword arguments to construct a layer.

    Input shape: A numeric `Tensor`, `SparseTensor` or `RaggedTensor` of shape
        `[batch_size, d1, ..., dm]`

    Output shape: An int64 tensor of shape `[batch_size, d1, ..., dm]`

    """

    def __init__(self, num_buckets, default_value=0):
        super(RoundIdentity, self).__init__()
        self.num_buckets = num_buckets
        self.default_value = default_value

    def call(self, inputs):
        if isinstance(inputs, tf.SparseTensor):
            id_values = self._round_and_truncate(inputs.values)
            result = tf.SparseTensor(
                indices=inputs.indices,
                values=id_values,
                dense_shape=inputs.dense_shape,
            )
        elif ragged_tensor.is_ragged(inputs):
            result = ragged_functional_ops.map_flat_values(
                self._round_and_truncate, inputs
            )
        else:
            result = self._round_and_truncate(inputs)
        return tf.cast(result, tf.int64)

    def _round_and_truncate(self, values):
        values = tf.keras.backend.round(values)
        values = tf.cast(values, tf.int64)
        num_buckets = tf.cast(self.num_buckets, tf.int64)
        default_value = tf.cast(self.default_value, tf.int64)
        values = tf.where(
            tf.logical_or(values < 0, values >= num_buckets),
            x=tf.fill(dims=tf.shape(values), value=default_value),
            y=values,
        )
        return values

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "num_buckets": self.num_buckets,
            "default_value": self.default_value,
        }
        base_config = super(RoundIdentity, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
