import collections

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, math_ops, sparse_ops

EmbeddingAndIds = collections.namedtuple(
    "EmbeddingAndIds", ["batch_embedding", "batch_ids"]
)


class EmbeddingDelegate(object):
    """
    The common component to interact the external embedding
    storage such as the parameter server.
    Both ElasticDL Embedding Layer and Embedding Column will
    use this component.
    """

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
        self.tape = None

    def set_tape(self, tape):
        self.tape = tape

    def init_for_graph_mode_if_necessary(self):
        if (
            tf.executing_eagerly()
            or self._embedding_and_ids_graph
            or not self.tape
        ):
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

    def embedding_lookup(self, ids):
        self.init_for_graph_mode_if_necessary()

        ids = tf.cast(ids, tf.int64)

        ids = tf.convert_to_tensor(ids, name=self.name + "_ids")
        flat_ids = tf.reshape(ids, [-1])
        unique_ids, idx = tf.unique(flat_ids)

        # There is a memory leak when using tf.py_function with eager
        # execution. So, we only use tf.py_function in graph mode.
        if isinstance(unique_ids, ops.EagerTensor):
            batch_embedding = self.gather_embedding_vectors(unique_ids)
            batch_embedding = tf.constant(batch_embedding, dtype=tf.float32)
        else:
            batch_embedding = tf.py_function(
                self.gather_embedding_vectors,
                inp=[unique_ids],
                Tout=tf.float32,
            )

        # Gradient for `batch_embedding` is SparseTensor here due to
        # `tf.gather` op. `tf.gather` accesses tensor slices, resulting in
        # sparse tensor gradient.
        # TODO: use tf.cond rather than python if statement
        if self.tape:
            batch_embedding = self.record_gradients(batch_embedding, flat_ids)

        result = tf.gather(batch_embedding, idx)
        # tf.reshape does not support shape with None. Replace None with -1.
        if ids.get_shape().rank == 2:
            input_length = ids.get_shape()[1]
            if input_length is None:
                result.set_shape(shape=(None, None, self.output_dim))
                return result
            output_shape = (-1, input_length, self.output_dim)
        else:
            output_shape = ids.get_shape().concatenate(self.output_dim)
        result = tf.reshape(result, output_shape)
        return result

    def safe_embedding_lookup_sparse(
        self, sparse_ids, sparse_weights=None, combiner="mean", default_id=None
    ):

        sparse_ids = _prune_invalid_ids(sparse_ids)
        # Fill in dummy values for empty features, if necessary.
        sparse_ids, is_row_empty = sparse_ops.sparse_fill_empty_rows(
            sparse_ids, 0
        )
        unique_ids, idx = tf.unique(sparse_ids.values)

        segment_ids = sparse_ids.indices[:, 0]
        if segment_ids.dtype != tf.int32:
            segment_ids = tf.cast(segment_ids, tf.int32)

        ids = sparse_ids.values
        unique_ids, idx = tf.unique(ids)

        if isinstance(unique_ids, ops.EagerTensor):
            batch_embedding = self.gather_embedding_vectors(unique_ids)
            batch_embedding = tf.constant(batch_embedding, dtype=tf.float32)
        else:
            batch_embedding = tf.py_function(
                self.gather_embedding_vectors,
                inp=[unique_ids],
                Tout=tf.float32,
            )

        if sparse_weights is not None:
            if self.tape:
                batch_embedding = self.record_gradients(
                    tape=self.tape, batch_embedding=batch_embedding, ids=ids
                )

            weights = sparse_weights.values
            if weights.dtype != batch_embedding.dtype:
                weights = math_ops.cast(weights, batch_embedding.dtype)

            batch_embedding = array_ops.gather(batch_embedding, idx)

            # Reshape weights to allow broadcast
            ones = array_ops.fill(
                array_ops.expand_dims(array_ops.rank(batch_embedding) - 1, 0),
                1,
            )
            bcast_weights_shape = array_ops.concat(
                [array_ops.shape(weights), ones], 0
            )

            orig_weights_shape = weights.get_shape()
            weights = array_ops.reshape(weights, bcast_weights_shape)

            # Set the weight shape, since after reshaping to
            # bcast_weights_shape, the shape becomes None.
            if batch_embedding.get_shape().ndims is not None:
                weights.set_shape(
                    orig_weights_shape.concatenate(
                        [
                            1
                            for _ in range(
                                batch_embedding.get_shape().ndims - 1
                            )
                        ]
                    )
                )

            batch_embedding *= weights

            if combiner == "sum":
                batch_embedding = math_ops.segment_sum(
                    batch_embedding, segment_ids
                )
            elif combiner == "mean":
                batch_embedding = math_ops.segment_sum(
                    batch_embedding, segment_ids
                )
                weight_sum = math_ops.segment_sum(weights, segment_ids)
                batch_embedding = math_ops.div(batch_embedding, weight_sum)
            elif combiner == "sqrtn":
                batch_embedding = math_ops.segment_sum(
                    batch_embedding, segment_ids
                )
                weights_squared = math_ops.pow(weights, 2)
                weight_sum = math_ops.segment_sum(weights_squared, segment_ids)
                weight_sum_sqrt = math_ops.sqrt(weight_sum)
                batch_embedding = math_ops.div(
                    batch_embedding, weight_sum_sqrt
                )
            else:
                assert False, "Unrecognized combiner"
        else:
            if self.tape:
                batch_embedding = self.record_gradients(
                    tape=self.tape,
                    batch_embedding=batch_embedding,
                    ids=unique_ids,
                )

            assert idx is not None
            if combiner == "sum":
                batch_embedding = math_ops.sparse_segment_sum(
                    batch_embedding, idx, segment_ids
                )
            elif combiner == "mean":
                batch_embedding = math_ops.sparse_segment_mean(
                    batch_embedding, idx, segment_ids
                )
            elif combiner == "sqrtn":
                batch_embedding = math_ops.sparse_segment_sqrt_n(
                    batch_embedding, idx, segment_ids
                )
            else:
                assert False, "Unrecognized combiner"

        # Broadcast is_row_empty to the same shape as embedding_lookup_result,
        # for use in Select.
        is_row_empty = array_ops.tile(
            array_ops.reshape(is_row_empty, [-1, 1]),
            array_ops.stack([1, array_ops.shape(batch_embedding)[1]]),
        )

        batch_embedding = array_ops.where(
            is_row_empty,
            array_ops.zeros_like(batch_embedding),
            batch_embedding,
            name=self.name,
        )
        batch_embedding.set_shape((None, self.output_dim))
        return batch_embedding

    def gather_embedding_vectors(self, unique_ids):
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
        self.tape = None
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


def _prune_invalid_ids(sparse_ids):
    """Prune invalid IDs (< 0) from the input ids."""
    is_id_valid = tf.greater_equal(sparse_ids.values, 0)
    sparse_ids = sparse_ops.sparse_retain(sparse_ids, is_id_valid)
    return sparse_ids
