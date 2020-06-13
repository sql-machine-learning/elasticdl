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
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.ops import embedding_ops, math_ops


class SparseEmbedding(tf.keras.layers.Layer):
    """
    Input: indexes for the embedding entries with a shape of
        (batch_size, input_length). Input is a SparseTensor.
    Output:
        embeddings with a shape (batch_size, output_dim)
    Arguments:
        input_dim: the max input id. If 0 or None, will not check the range of
            input embedding ids.
        output_dim: the dimension of the embedding vector
        embeddings_initializer: Initializer for embedding table.
        combiner: A string specifying the reduction op or None if not used.
            "mean", "sqrtn" and "sum" are supported for the reduction op.
            If input is SparseTensor, combiner must set as a reduction op.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        embeddings_initializer="uniform",
        combiner="mean",
        **kwargs
    ):
        dtype = kwargs.pop("dtype", K.floatx())
        super(SparseEmbedding, self).__init__(dtype=dtype, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.combiner = combiner

        if self.combiner not in ["sum", "mean", "sqrtn"]:
            raise ValueError(
                "combiner must set sum, mean or sqrtn for sparse input"
            )

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name="embeddings",
            trainable=True,
        )
        self.built = True

    def call(self, inputs):
        # When saving a model with the layer, `tf.saved_model.save` will
        # feed the inputs with a Tensor not a SparseTensor, so we should
        # convert Tensor to `SparseTensor`.
        if not isinstance(inputs, tf.SparseTensor):
            idx = tf.where(tf.not_equal(inputs, 0))
            inputs = tf.SparseTensor(
                idx, tf.gather_nd(inputs, idx), (-1, self.input_dim)
            )

        dtype = K.dtype(inputs)
        if dtype != "int32" and dtype != "int64":
            inputs = math_ops.cast(inputs, "int32")
        out = embedding_ops.safe_embedding_lookup_sparse(
            embedding_weights=self.embeddings,
            sparse_ids=inputs,
            sparse_weights=None,
            combiner=self.combiner,
        )
        return out
