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


class ConcatenateWithOffset(tf.keras.layers.Layer):
    """Layer that add offset to each id tensor in the input list and
    then concatenate these tensors.

    It takes as input a list of tensors and returns a single tensor.
    Firstly, it will add an offset in offsets for each tensor in inputs.
    Then concatenate them to a single tensor. The tensor in inputs
    must have the same type, `Tensor` or `RaggedTensor` or `SparseTensor` and
    the same shape.

    Example :
    ```python
        a1 = tf.constant([[1], [1], [1]])
        a2 = tf.constant([[2], [2], [2]])
        offsets = [0, 10]
        layer = ConcatenateWithOffset(offsets=offsets, axis=1)
        layer([a1, a2])
        [[ 1 12]
         [ 1 12]
         [ 1 12]]
    ```

    Arguments:
        offsets: numeric list to add
        axis: Axis along which to concatenate.
        **kwargs: standard layer keyword arguments.
    """

    def __init__(self, offsets, axis=-1):
        super(ConcatenateWithOffset, self).__init__()
        self.offsets = offsets
        self.axis = axis

    def call(self, inputs):
        if self.offsets is None:
            return self._call_with_none_offsets(inputs)

        return self._call_with_offsets(inputs)

    def _call_with_offsets(self, inputs):
        ids_with_offset = []
        if not isinstance(inputs, list):
            return inputs

        if len(self.offsets) != len(inputs):
            raise ValueError(
                "The offsets length is not equal to inputs length"
                "the inputs are {}, offsets are {}".format(
                    inputs, self.offsets
                )
            )
        for i, tensor in enumerate(inputs):
            if isinstance(tensor, tf.SparseTensor):
                ids_with_offset.append(
                    tf.SparseTensor(
                        indices=tensor.indices,
                        values=tensor.values + self.offsets[i],
                        dense_shape=tensor.dense_shape,
                    )
                )
            else:
                ids_with_offset.append(tensor + self.offsets[i])

        if isinstance(ids_with_offset[0], tf.SparseTensor):
            result = tf.sparse.concat(
                axis=self.axis, sp_inputs=ids_with_offset
            )
        else:
            result = tf.concat(ids_with_offset, axis=self.axis)

        return result

    def _call_with_none_offsets(self, inputs):
        if isinstance(inputs[0], tf.SparseTensor):
            result = tf.sparse.concat(axis=self.axis, sp_inputs=inputs)
        else:
            result = tf.concat(inputs, axis=self.axis)

        return result
