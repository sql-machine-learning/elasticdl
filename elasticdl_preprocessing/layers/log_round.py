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


def log(x, base):
    x = tf.cast(x, tf.float64)
    numerator = tf.math.log(x)

    if base is None:
        return numerator

    denominator = tf.math.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator


class LogRound(tf.keras.layers.Layer):
    """Cast a numeric value into a discrete integer value by
    `round(log(x))`.


    Example :
    ```python
        layer = LogRound(num_bins=16, base=2)
        inp = np.asarray([[1.2], [1.6], [0.2], [3.1], [100]])
        layer(inp)
        [[0], [1], [0], [2], [7]]
    ```

    Arguments:
        num_bins: Range of inputs and outputs is `[0, num_bins)`.
        **kwargs: Keyword arguments to construct a layer.

    Input shape: A numeric `Tensor`, `SparseTensor` or `RaggedTensor` of shape
        `[batch_size, d1, ..., dm]`

    Output shape: An int64 tensor of shape `[batch_size, d1, ..., dm]`

    """

    def __init__(self, num_bins, default_value=0, base=None):
        super(LogRound, self).__init__()
        self.num_bins = num_bins
        self.default_value = default_value
        self.base = base

    def call(self, inputs):
        if isinstance(inputs, tf.SparseTensor):
            id_values = self._log_round(inputs.values)
            result = tf.SparseTensor(
                indices=inputs.indices,
                values=id_values,
                dense_shape=inputs.dense_shape,
            )
        elif ragged_tensor.is_ragged(inputs):
            result = ragged_functional_ops.map_flat_values(
                self._log_round, inputs
            )
        else:
            result = self._log_round(inputs)
        return tf.cast(result, tf.int64)

    def _log_round(self, values):
        values = tf.cast(values, tf.float64)
        values = tf.math.round(log(values, base=self.base))
        values = tf.cast(values, tf.int64)
        num_bins = tf.cast(self.num_bins, tf.int64)
        default_value = tf.cast(self.default_value, tf.int64)
        values = tf.where(
            tf.logical_or(values < 0, values >= num_bins),
            x=tf.fill(dims=tf.shape(values), value=default_value),
            y=values,
        )
        return values

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "num_bins": self.num_bins,
            "base": self.base,
            "default_value": self.default_value,
        }
        base_config = super(LogRound, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
