import collections
import math

import tensorflow as tf
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.feature_column import feature_column_v2 as fc_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import init_ops


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

        unique_ids, idx = tf.unique(sparse_ids.values)
        batch_embedding = tf.py_function(
            self.lookup_embedding, inp=[unique_ids], Tout=tf.float32
        )

        segment_ids = sparse_ids.indices[:, 0]
        if segment_ids.dtype != tf.int32:
            segment_ids = tf.cast(segment_ids, tf.int32)

        # TODO(brightcoder01): Add combine with sparse_weights
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

        return batch_embedding

    def lookup_embedding(self, unique_ids):
        raise Exception("Not implemented yet")

    def set_tape(self, tape):
        self.tape = tape

    def reset(self):
        self.tape = None
