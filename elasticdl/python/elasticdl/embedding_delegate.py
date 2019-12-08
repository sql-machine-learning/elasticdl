import collections

import numpy as np
import tensorflow as tf

EmbeddingAndIds = collections.namedtuple(
    "EmbeddingAndIds", ["batch_embedding", "batch_ids"]
)


class EmbeddingDelegate(object):
    '''
    The common component to interact the external embedding
    storage such as the parameter server.
    Both ElasticDL Embedding Layer and Embedding Column will
    use this component.
    '''
    def __init__(self, input_dim, output_dim, name):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self._lookup_embedding_func = None
        self._embedding_and_ids_eagerly = []
        # BET's shape and ids' shape in `self._embedding_and_ids_graph` have
        # `None` dimension. This is because they have different shapes in
        # different iterations.
        # `tf.Variable` requires initial value if shape has `None` dimension.
        self._embedding_and_ids_graph = []

    def init_for_graph_mode_if_necessary(self):
        if tf.executing_eagerly() or self._embedding_and_ids_graph:
            return

        self._embedding_and_ids_graph = [
            EmbeddingAndIds(
                batch_embedding=tf.Variable(
                    # In some cases, `tf.Variable` requires that initial value
                    # is callable.
                    initial_value=lambda: tf.zeros((1, self.output_dim)),
                    shape=tf.TensorShape((None, self.output_dim)),
                    dtype=tf.float32,
                    trainable=True,
                ),
                batch_ids=tf.Variable(
                    initial_value=lambda: tf.zeros((1, 1), dtype=tf.int64),
                    shape=tf.TensorShape(None),
                    dtype=tf.int64,
                    trainable=False,
                ),
            )
        ]

    def lookup_embedding(self, unique_ids):
        ids = unique_ids.numpy()
        self._check_id_valid(ids)
        if self._lookup_embedding_func:
            embedding_vectors = self._lookup_embedding_func(self.name, ids)
            return embedding_vectors

    def _check_id_valid(self, ids):
        if not self.input_dim:
            return

        first_may_exceed_id = ids[np.argmax(ids >= self.input_dim)]
        if self.input_dim is not None and first_may_exceed_id > self.input_dim:
            raise ValueError(
                " The embedding id cannot be bigger "
                "than input_dim. id = %d is not in [0, %d)"
                % (first_may_exceed_id, self.input_dim)
            )

    def record_gradients(self, tape, batch_embedding, ids):
        if tf.executing_eagerly():
            tape.watch(batch_embedding)
            self._embedding_and_ids_eagerly.append(
                EmbeddingAndIds(batch_embedding, ids)
            )
        else:
            # In graph mode, assigning tensors to trainable variables is
            # allowed and tape can record the gradients of trainable
            # variables automatically.
            embedding_and_ids = self._embedding_and_ids_graph[0]
            embedding_and_ids.batch_embedding.assign(batch_embedding)
            embedding_and_ids.batch_ids.assign(ids)
            batch_embedding = embedding_and_ids.batch_embedding

        return batch_embedding

    def reset(self):
        self._embedding_and_ids_eagerly = []

    def set_lookup_embedding_func(self, lookup_embedding_func):
        self._lookup_embedding_func = lookup_embedding_func

    @property
    def embedding_and_ids(self):
        """
        Return bet and ids pairs.
        """
        if self._embedding_and_ids_eagerly:
            return self._embedding_and_ids_eagerly
        return self._embedding_and_ids_graph
