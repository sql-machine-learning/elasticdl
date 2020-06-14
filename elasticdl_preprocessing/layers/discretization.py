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

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.ops import math_ops


class Discretization(tf.keras.layers.Layer):
    """Buckets data into discrete ranges.

    TensorFlow 2.2 has developed `tf.keras.layers.preprocessing.Discretization`
    but not released it yet. So the layer is a simple temporary version
    `tensorflow.python.keras.layers.preprocessing.discretization.Discretization`

    Input shape:
        Any `tf.Tensor` or `tf.RaggedTensor` of dimension 2 or higher.

    Output shape:
        The same as the input shape with tf.int64.

    Attributes:
        bins: Optional boundary specification. Bins include the left boundary
            and exclude the right boundary, so `bins=[0., 1., 2.]` generates
            bins `(-inf, 0.)`, `[0., 1.)`, `[1., 2.)`, and `[2., +inf)`.
  """

    def __init__(self, bins, **kwargs):
        super(Discretization, self).__init__(**kwargs)
        self._supports_ragged_inputs = True
        self.bins = bins

    def num_bins(self):
        """The bins is a list with boundaries, so the number of bins is
        len(bins) + 1.
        """
        return len(self.bins) + 1

    def get_config(self):
        config = {
            "bins": self.bins,
        }
        base_config = super(Discretization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        if isinstance(inputs, tf.RaggedTensor):
            integer_buckets = tf.ragged.map_flat_values(
                math_ops._bucketize, inputs, boundaries=self.bins
            )
            integer_buckets = tf.identity(integer_buckets)
        elif isinstance(inputs, tf.SparseTensor):
            integer_bucket_values = math_ops._bucketize(
                inputs.values, boundaries=self.bins
            )
            integer_buckets = tf.SparseTensor(
                indices=inputs.indices,
                values=integer_bucket_values,
                dense_shape=inputs.dense_shape,
            )
        else:
            integer_buckets = math_ops._bucketize(inputs, boundaries=self.bins)

        return tf.cast(integer_buckets, tf.int64)
