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

_NUMBER_DTYPES = [
    tf.int8,
    tf.uint8,
    tf.int16,
    tf.uint16,
    tf.int32,
    tf.uint32,
    tf.int64,
    tf.uint64,
    tf.float16,
    tf.float32,
    tf.float64,
    tf.bfloat16,
    tf.double,
]


class ToNumber(tf.keras.layers.Layer):
    """Convert the inputs to a number dtype (int, float, double)

    Input Shape: Tensor or SparseTensor of any shape
    Output Shape: Tensor or SparseTensor of the same shape with input
    """

    def __init__(self, out_type, default_value):
        super(ToNumber, self).__init__()
        if out_type not in _NUMBER_DTYPES:
            raise ValueError("{} is not a number type.".format(out_type))
        self.out_type = out_type
        self.default_value = default_value

    def call(self, inputs):
        if isinstance(inputs, tf.SparseTensor):
            number_value = self._cast_dense_to_number(inputs.values)
            return tf.SparseTensor(
                indices=inputs.indices,
                values=number_value,
                dense_shape=inputs.dense_shape,
            )
        else:
            return self._cast_dense_to_number(inputs)

    def _cast_dense_to_number(self, dense_inputs):
        if dense_inputs.dtype is tf.string:
            default_value = str(self.default_value)
            outputs = tf.where(
                tf.equal(dense_inputs, ""), x=default_value, y=dense_inputs
            )
            outputs = tf.strings.to_number(outputs, out_type=self.out_type)
        else:
            outputs = tf.cast(dense_inputs, self.out_type)

        return outputs
