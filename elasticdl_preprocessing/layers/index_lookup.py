from __future__ import absolute_import, division, print_function

import collections

import tensorflow as tf
from tensorflow.python.ops import lookup_ops


class IndexLookup(tf.keras.layers.Layer):
    """Maps strings from a vocabulary to integer indices.

    This layer transforms categorical inputs to hashed output. It converts a
    sequence of int or string to a sequence of int. TensorFlow 2.2 has
    developed `tf.keras.layers.preprocessing.Hashing` but not released it yet.
    So the layer is a simple temporary version.
    https://github.com/tensorflow/tensorflow/blob/r2.2/tensorflow/python/keras/layers/preprocessing/hashing.py

    Example:
    ```python
    layer = IndexLookup(vocabulary=['A', 'B', 'C'])
    inp = np.asarray([['A'], ['B'], ['C'], ['D'], ['E']])
    layer(inputs)
    [[0], [1], [2], [3], [3]]
    ```

    Attributes:
    num_oov_tokens: The number of out-of-vocabulary tokens to use; defaults to
        1. If this value is more than 1, OOV inputs are hashed to determine
        their OOV value; if this value is 0, passing an OOV input will result
        in a '-1' being returned for that value in the output tensor.
    vocabulary: An optional list of vocabulary terms, or a path to a text file
        containing a vocabulary to load into this layer. The file should
        contain one token per line.

    Input shape: A string tensor of shape
        `[batch_size, d1, ..., dm]`. The tensor can be `tf.Tensor`,
        `tf.SparseTensor` and `tf.RaggedTensor`

    Output shape: An int64 tensor of shape `[batch_size, d1, ..., dm]`

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
        self.table = lookup_ops.index_table_from_tensor(
            vocabulary_list=self.vocabulary,
            num_oov_buckets=self.num_oov_tokens,
        )

    def call(self, inputs):
        if isinstance(inputs, tf.SparseTensor):
            lookup_id = self.table.lookup(inputs.values)
            output = tf.SparseTensor(
                indices=inputs.indices,
                values=lookup_id,
                dense_shape=inputs.dense_shape,
            )
        elif isinstance(inputs, tf.RaggedTensor):
            return tf.ragged.map_flat_values(self.table.lookup, inputs,)
        else:
            output = self.table.lookup(inputs)
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
