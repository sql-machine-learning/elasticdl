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


def lookup_embedding_func(
    id_tensors, max_ids, embedding_dim,
):
    """
    Args:
        id_layers: dict, the key is a string and
            the value is tf.keras.Input.
        standardized_tensor:
        input_tensors: dict, the key is a string and the value
            is a tensor outputed by the transform function
        max_ids: dict, the key is a string and the value is the max
            integer id of this group.
        deep_embedding_dim: The output dimension of embedding layer for
            deep parts.
    """

    embeddings = []
    for name, id_tensor in id_tensors.items():
        wide_embedding_layer = tf.keras.layers.Embedding(
            max_ids[name], embedding_dim
        )
        embedding = wide_embedding_layer(id_tensor)
        embedding_sum = tf.keras.backend.sum(embedding, axis=1)
        embeddings.append(embedding_sum)
    return embeddings


class DNN(tf.keras.layers.Layer):
    """ DNN layer

    Args:
        hidden_units: A list with integers
        activation: activation function name, like "relu"
    inputs: A Tensor
    """

    def __init__(self, hidden_units, activation=None, **kwargs):
        super(DNN, self).__init__(**kwargs)
        self.dense_layers = []
        if not hidden_units:
            raise ValueError("The hidden units cannot be empty")
        for hidden_unit in hidden_units:
            self.dense_layers.append(
                tf.keras.layers.Dense(hidden_unit, activation=activation)
            )

    def call(self, inputs):
        output = inputs
        for layer in self.dense_layers:
            output = layer(output)
        return output
