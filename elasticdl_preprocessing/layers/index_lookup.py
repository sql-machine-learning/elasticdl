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

import collections

import tensorflow as tf
from tensorflow.python.ops import lookup_ops


class IndexLookup(tf.keras.layers.Layer):
    """Maps strings to integer indices by looking up a vocabulary.

    This layer transforms categorical inputs to zero-based integer by
    lookuping with a vocabulary list. TensorFlow 2.2 has developed
    `tf.keras.layers.preprocessing.IndexLookup` but not released it yet.
    So the layer is a simple temporary version. The codes in TensorFlow 2.2 is
    `tensorflow.python.keras.layers.preprocessing.index_lookup.IndexLookup`.

    Note that the TensorFlow version with the layer must be greater than 2.0.0.

    Example:
    ```python
    layer = IndexLookup(vocabulary=['A', 'B', 'C'])
    inp = np.array([['A'], ['B'], ['C'], ['D'], ['E']])
    layer(inputs)
    ```
    Then output will be `[[0], [1], [2], [3], [3]]`

    Attributes:
    num_oov_tokens: The number of out-of-vocabulary tokens to use; defaults to
        1. If this value is more than 1,
        `hash(inputs) % num_oov_tokens + len(vocabulary)` converts OOV inputs
        to integer values.
    vocabulary: A list of vocabulary terms, or a path to a text file
        containing a vocabulary to load into this layer. The file should
        contain one token per line.

    Input: A string `tf.Tensor`,`tf.SparseTensor` or
        `tf.RaggedTensor`.

    Output: An int64 tensor with the same type as input.

    """

    def __init__(self, vocabulary=None, num_oov_tokens=1, **kwargs):
        super(IndexLookup, self).__init__()
        self.num_oov_tokens = num_oov_tokens

        if vocabulary is not None and isinstance(vocabulary, str):
            vocabulary = self._get_vocabulary_from_file(vocabulary)
            vocabulary_set = set(vocabulary)
            if len(vocabulary) != len(vocabulary_set):
                repeated_items = [
                    item
                    for item, count in collections.Counter(vocabulary).items()
                    if count > 1
                ]
                raise ValueError(
                    "The passed vocabulary has at least one repeated "
                    "term. Please uniquify your dataset before passing "
                    "it to IndexLookup(). The repeated terms are %s"
                    % repeated_items
                )
        self.vocabulary = vocabulary

    def build(self, input_shape):
        self._table = lookup_ops.index_table_from_tensor(
            vocabulary_list=self.vocabulary,
            num_oov_buckets=self.num_oov_tokens,
        )

    def call(self, inputs):
        if isinstance(inputs, tf.SparseTensor):
            lookup_id = self._table.lookup(inputs.values)
            output = tf.SparseTensor(
                indices=inputs.indices,
                values=lookup_id,
                dense_shape=inputs.dense_shape,
            )
        elif isinstance(inputs, tf.RaggedTensor):
            return tf.ragged.map_flat_values(self._table.lookup, inputs,)
        else:
            output = self._table.lookup(inputs)
        return tf.cast(output, tf.int64)

    def _get_vocabulary_from_file(self, vocabulary_path):
        vocab = []
        with tf.io.gfile.GFile(vocabulary_path, "r") as reader:
            while True:
                # Get the next line, and break if it is None.
                text = reader.readline()
                if not text:
                    break

                # Convert the raw text into UTF8 and strip whitespace.
                if isinstance(text, str):
                    token = text
                elif isinstance(text, bytes):
                    token = text.decode("utf-8", "ignore")
                token = token.strip()
                vocab.append(token)
        return vocab

    def vocab_size(self):
        return self._table.size().numpy()

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "num_oov_tokens": self.num_oov_tokens,
            "vocabulary": None,
        }
        base_config = super(IndexLookup, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
