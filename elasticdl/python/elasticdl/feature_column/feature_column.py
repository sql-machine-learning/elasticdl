import collections
import math

import numpy as np
import tensorflow as tf
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.feature_column import feature_column_v2 as fc_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops, init_ops, math_ops

from elasticdl.python.common.embedding import EmbeddingAndIds


def embedding_column(
    categorical_column,
    dimension,
    combiner="mean",
    initializer=None,
    max_norm=None,
    trainable=True,
):
    """
    Create a customized EmbeddingColumn for ElasticDL.
    The native EmbeddingColumn will create a variable to
    store the entire embedding table. It can't leverage the
    benefit from the ElasticDL parameter server to partition
    the embedding table. Create this ElasticDL EmbeddingColumn
    to interact with ElasticDL parameter server.
    The API signature is based on the native
    tf.feature_column.embedding_column and
    remove some unused parameters.

    Args:
      categorical_column: A `CategoricalColumn` created by a
        `categorical_column_with_*` function. This column produces
        the sparse IDs that are inputs to the embedding lookup.
      dimension: An integer specifying dimension of the embedding, must be > 0.
      combiner: A string specifying how to reduce if there are multiple entries
        in a single row. Currently 'mean', 'sqrtn' and 'sum' are supported,
        with 'mean' the default. 'sqrtn' often achieves good accuracy, in
        particular with bag-of-words columns. Each of this can be thought as
        example level normalizations on the column. For more information, see
        `tf.embedding_lookup_sparse`.
      initializer: A variable initializer function to be used in embedding
        variable initialization. If not specified, defaults to
        `truncated_normal_initializer` with mean `0.0` and
        standard deviation `1/sqrt(dimension)`.
      max_norm: If not `None`, embedding values are l2-normalized
        to this value.
      trainable: Whether or not the embedding is trainable. Default is True.

    Returns:
        `DenseColumn` that converts from sparse input.

    Raises:
        ValueError: if `dimension` not > 0.
        ValueError: if `initializer` is specified and is not callable.
    """
    if (dimension is None) or (dimension < 1):
        raise ValueError("Invalid dimension {}.".format(dimension))

    if (initializer is not None) and (not callable(initializer)):
        raise ValueError(
            "initializer must be callable if specified. "
            "Embedding of column_name: {}".format(categorical_column.name)
        )
    if initializer is None:
        initializer = init_ops.truncated_normal_initializer(
            mean=0.0, stddev=1 / math.sqrt(dimension)
        )

    return EmbeddingColumn(
        categorical_column=categorical_column,
        dimension=dimension,
        combiner=combiner,
        initializer=initializer,
        max_norm=max_norm,
        trainable=trainable,
    )


class EmbeddingColumn(
    fc_lib.DenseColumn,
    fc_lib.SequenceDenseColumn,
    fc_old._DenseColumn,
    fc_old._SequenceDenseColumn,
    collections.namedtuple(
        "EmbeddingColumn",
        (
            "categorical_column",
            "dimension",
            "combiner",
            "initializer",
            "max_norm",
            "trainable",
        ),
    ),
):
    def __init__(self, **kwargs):
        self.tape = None
        self._lookup_embedding_func = None
        self._embedding_and_ids_eagerly = []

        default_num_buckets = (
            self.categorical_column.num_buckets
            if self._is_v2_column
            else self.categorical_column._num_buckets
        )  # pylint: disable=protected-access
        self.num_buckets = getattr(
            self.categorical_column, "num_buckets", default_num_buckets
        )

    @property
    def _is_v2_column(self):
        return (
            isinstance(self.categorical_column, fc_lib.FeatureColumn)
            and self.categorical_column._is_v2_column
        )

    @property
    def name(self):
        """See `FeatureColumn` base class."""
        return "{}_embedding_elasticdl".format(self.categorical_column.name)

    @property
    def parse_example_spec(self):
        """See `FeatureColumn` base class."""
        return self.categorical_column.parse_example_spec

    @property
    def variable_shape(self):
        """See `DenseColumn` base class."""
        return tensor_shape.TensorShape([self.dimension])

    def get_dense_tensor(self, transformation_cache, state_manager):
        if isinstance(
            self.categorical_column, fc_lib.SequenceCategoricalColumn
        ):
            raise ValueError(
                "In embedding_column: {}. "
                "categorical_column must not be of "
                "type SequenceCategoricalColumn. "
                "Suggested fix A: If you wish to use DenseFeatures, use a "
                "non-sequence categorical_column_with_*. "
                "Suggested fix B: If you wish to create sequence input, use "
                "SequenceFeatures instead of DenseFeatures. "
                "Given (type {}): {}".format(
                    self.name,
                    type(self.categorical_column),
                    self.categorical_column,
                )
            )
        # Get sparse IDs and weights.
        sparse_tensors = self.categorical_column.get_sparse_tensors(
            transformation_cache, state_manager
        )

        # Look up the embedding from the sparse input
        sparse_ids = sparse_tensors.id_tensor
        sparse_weights = sparse_tensors.weight_tensor

        unique_ids, idx = tf.unique(sparse_ids.values)
        batch_embedding = tf.py_function(
            self.lookup_embedding, inp=[unique_ids], Tout=tf.float32
        )

        if self.tape:
            batch_embedding = self._record_gradients(
                batch_embedding, unique_ids
            )

        segment_ids = sparse_ids.indices[:, 0]
        if segment_ids.dtype != tf.int32:
            segment_ids = tf.cast(segment_ids, tf.int32)

        if sparse_weights is not None:
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
            weights = array_ops.reshape(weights, bcast_weights_shape)

            batch_embedding *= weights

            if self.combiner == "sum":
                batch_embedding = math_ops.segment_sum(
                    batch_embedding, segment_ids
                )
            elif self.combiner == "mean":
                batch_embedding = math_ops.segment_sum(
                    batch_embedding, segment_ids
                )
                weight_sum = math_ops.segment_sum(weights, segment_ids)
                batch_embedding = math_ops.div(batch_embedding, weight_sum)
            elif self.combiner == "sqrtn":
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
            assert idx is not None
            if self.combiner == "sum":
                batch_embedding = tf.sparse.segment_sum(
                    batch_embedding, idx, segment_ids
                )
            elif self.combiner == "mean":
                batch_embedding = tf.sparse.segment_mean(
                    batch_embedding, idx, segment_ids
                )
            elif self.combiner == "sqrtn":
                batch_embedding = tf.sparse.segment_sqrt_n(
                    batch_embedding, idx, segment_ids
                )
            else:
                assert False, "Unrecognized combiner"

        return batch_embedding

    def _record_gradients(self, batch_embedding, ids):
        self.tape.watch(batch_embedding)
        self._embedding_and_ids_eagerly.append(
            EmbeddingAndIds(batch_embedding, ids)
        )

        return batch_embedding

    def lookup_embedding(self, unique_ids):
        ids = unique_ids.numpy()
        self._check_id_valid(ids)
        if self._lookup_embedding_func:
            embedding_vectors = self._lookup_embedding_func(self._name, ids)
            return embedding_vectors

    def _check_id_valid(self, ids):
        if not self.num_buckets:
            return
        first_may_exceed_id = ids[np.argmax(ids >= self.num_buckets)]
        if self.num_buckets is not None and first_may_exceed_id > self.num_buckets:
            raise ValueError(
                " The embedding id cannot be bigger "
                "than num_buckets. id = %d is not in [0, %d)"
                % (first_may_exceed_id, self.num_buckets)
            )

    def set_tape(self, tape):
        self.tape = tape

    def set_lookup_embedding_func(self, func):
        """Sets function for looking up embeddings in the PS.

        Args:
            func: The function used to look up embeddings. The arguments of
                are `(layer_name, embedding_id_list)`, where `layer_name` is
                the name of embedding layer, and `embedding_id_list` is a list
                of embedding ids to be looked up.
        """
        self._lookup_embedding_func = func

    def reset(self):
        self.tape = None
        self._embedding_and_ids_eagerly = []

    @property
    def embedding_and_ids(self):
        return self._embedding_and_ids_eagerly
