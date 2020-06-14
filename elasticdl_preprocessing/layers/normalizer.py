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


class Normalizer(tf.keras.layers.Layer):
    """Normalize the numeric tensors by (x-subtractor)/divisor

    Example :
    ```python
        layer = Normalizer(subtractor=1.0, divisor=2.0)
        inp = np.asarray([[3.0], [5.0], [7.0]])
        layer(inp)
        [[1.0], [2.0], [3.0]]
    ```

    Arguments:
        subtractor: A float value.
        divisor: A float value.

    Input shape: A numeric `Tensor`, `SparseTensor` or `RaggedTensor` of shape
        `[batch_size, d1, ..., dm]`

    Output shape: An float64 tensor of shape `[batch_size, d1, ..., dm]`

    """

    def __init__(self, subtractor, divisor, **kwargs):
        super(Normalizer, self).__init__(**kwargs)
        self._supports_ragged_inputs = True
        self.subtractor = subtractor
        self.divisor = divisor

    def build(self, input_shape):
        if self.divisor == 0:
            raise ValueError("The divisor cannot be 0")

    def get_config(self):
        config = {"subtractor": self.subtractor, "divisor": self.divisor}
        base_config = super(Normalizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        if isinstance(inputs, tf.RaggedTensor):
            normalized_tensor = tf.ragged.map_flat_values(
                self._normalize_fn, inputs
            )
        elif isinstance(inputs, tf.SparseTensor):
            normalize_values = self._normalize_fn(inputs.values)
            normalized_tensor = tf.SparseTensor(
                indices=inputs.indices,
                values=normalize_values,
                dense_shape=inputs.dense_shape,
            )
        else:
            normalized_tensor = self._normalize_fn(inputs)

        return normalized_tensor

    def _normalize_fn(self, x):
        x = tf.cast(x, tf.float32)
        subtractor = tf.cast(self.subtractor, tf.float32)
        divisor = tf.cast(self.divisor, tf.float32)
        return (x - subtractor) / divisor
