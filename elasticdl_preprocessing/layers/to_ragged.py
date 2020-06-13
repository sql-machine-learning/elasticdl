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

_COMMA_SEP = ","


class ToRagged(tf.keras.layers.Layer):
    """Converts a `Tensor` or `RaggedTensor` to a `RaggedTensor`,
    dropping ignore_value cells. If the input's dtype is string, split
    the string elements to convert the input to `RaggedTensor` firstly.

    Note that the TensorFlow version with the layer must be greater than 2.0.0.

    Example (Integer):
    ```python
        layer = ToRagged()
        input_tensor = tf.constant([[1], [-1], [4]], tf.int64)
        out = layer(input_tensor)
        [[1], [], [4]]
    ```

    Example (String):
    ```python
        layer = ToRagged()
        input_tensor = tf.constant([["1", "2", "3"], ["4", "5"], [""]])
        out = layer(input_tensor)
    ```
    The expected output is `[["1", "2", "3"], ["4", "5"], []]`

    Arguments:
        sep: Valid if the input's dtype is string.
        ignore_value: Entries in inputs equal to this value will be
            absent from the output `RaggedTensor`. If `None`, default value of
            input's dtype will be used ('' for `str`, -1 for `int`).

    Input shape: A numeric or string `Tensor` or `RaggedTensor` of shape
        `[batch_size, d1, ..., dm]`

    Output shape: An `RaggedTensor` with the same shape as inputs
    """

    def __init__(self, sep=_COMMA_SEP, ignore_value=None):
        super(ToRagged, self).__init__()
        self.sep = sep
        self.ignore_value = ignore_value

    def call(self, inputs):
        if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            raise TypeError(
                "The inputs must be a Tensor or RaggedTensor and "
                "the type of inputs is {}".format(type(inputs))
            )

        if isinstance(inputs, tf.Tensor):
            inputs = tf.RaggedTensor.from_tensor(inputs)

        ignore_value = self._get_ignore_value(inputs.dtype)
        if ignore_value is None:
            return inputs
        else:
            return tf.ragged.boolean_mask(
                inputs, tf.not_equal(inputs, ignore_value)
            )

    def _get_ignore_value(self, input_dtype):
        ignore_value = self.ignore_value

        if ignore_value is None:
            if input_dtype == tf.string:
                ignore_value = ""
            elif input_dtype.is_integer:
                ignore_value = -1
            else:
                return None

        return tf.cast(ignore_value, input_dtype)
