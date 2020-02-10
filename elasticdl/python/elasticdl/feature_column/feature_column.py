import collections
import itertools
import math

import tensorflow as tf
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.feature_column import feature_column_v2 as fc_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import init_ops

from elasticdl.python.elasticdl.embedding_delegate import EmbeddingDelegate


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
        default_num_buckets = (
            self.categorical_column.num_buckets
            if self._is_v2_column
            else self.categorical_column._num_buckets
        )  # pylint: disable=protected-access
        num_buckets = getattr(
            self.categorical_column, "num_buckets", default_num_buckets
        )

        self._embedding_delegate = EmbeddingDelegate(
            input_dim=num_buckets, output_dim=self.dimension, name=self.name
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
        return "{}_embedding".format(self.categorical_column.name)

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
        result = self._embedding_delegate.safe_embedding_lookup_sparse(
            sparse_ids, sparse_weights=sparse_weights, combiner=self.combiner
        )
        return result

    def lookup_embedding(self, unique_ids):
        return self._embedding_delegate.lookup_embedding(unique_ids)

    def set_tape(self, tape):
        self._embedding_delegate.set_tape(tape)

    def set_lookup_embedding_func(self, func):
        """Sets function for looking up embeddings in the PS.

        Args:
            func: The function used to look up embeddings. The arguments of
                are `(column_name, embedding_id_list)`, where `column_name` is
                the name of embedding column, and `embedding_id_list` is a list
                of embedding ids to be looked up.
        """
        self._embedding_delegate.set_lookup_embedding_func(func)

    def reset(self):
        self._embedding_delegate.reset()

    @property
    def embedding_and_ids(self):
        return self._embedding_delegate.embedding_and_ids


def concat_column(categorical_columns):
    if not isinstance(categorical_columns, list):
        raise ValueError("categorical_columns should be a list")

    if not categorical_columns:
        raise ValueError("categorical_columns shouldn't be empty")

    for column in categorical_columns:
        if not isinstance(column, fc_lib.CategoricalColumn):
            raise ValueError(
                "Items of categorical_columns should be CategoricalColumn."
                " Given:{}".format(column)
            )

    return ConcatColumn(categorical_columns=tuple(categorical_columns))


class ConcatColumn(
    fc_lib.CategoricalColumn,
    fc_old._CategoricalColumn,
    collections.namedtuple("ConcatColumn", ("categorical_columns")),
):
    def __init__(self, **kwargs):
        # Calculate the offset tensor
        total_num_buckets = 0
        leaf_column_num_buckets = []
        for categorical_column in self.categorical_columns:
            leaf_column_num_buckets.append(categorical_column.num_buckets)
            total_num_buckets += categorical_column.num_buckets
        self.accumulated_offsets = list(
            itertools.accumulate([0] + leaf_column_num_buckets[:-1])
        )
        self.total_num_buckets = total_num_buckets

    @property
    def _is_v2_column(self):
        for categorical_column in self.categorical_columns:
            if not categorical_column._is_v2_column:
                return False

        return True

    @property
    def name(self):
        feature_names = []
        for categorical_column in self.categorical_columns:
            feature_names.append(categorical_column.name)

        return "_C_".join(sorted(feature_names))

    @property
    def num_buckets(self):
        return self.total_num_buckets

    @property
    def _num_buckets(self):
        return self.total_num_buckets

    def transform_feature(self, transformation_cache, state_manager):
        feature_tensors = []
        for categorical_column in self.categorical_columns:
            ids_and_weights = categorical_column.get_sparse_tensors(
                transformation_cache, state_manager
            )
            feature_tensors.append(ids_and_weights.id_tensor)

        feature_tensors_with_offset = []
        for index, offset in enumerate(self.accumulated_offsets):
            feature_tensor = feature_tensors[index]
            feature_tensor_with_offset = tf.SparseTensor(
                indices=feature_tensor.indices,
                values=tf.cast(
                    tf.add(feature_tensor.values, offset), tf.int64
                ),
                dense_shape=feature_tensor.dense_shape,
            )
            feature_tensors_with_offset.append(feature_tensor_with_offset)

        return tf.sparse.concat(axis=-1, sp_inputs=feature_tensors_with_offset)

    def get_sparse_tensors(self, transformation_cache, state_manager):
        return fc_lib.CategoricalColumn.IdWeightPair(
            transformation_cache.get(self, state_manager), None
        )

    @property
    def parents(self):
        return list(self.categorical_columns)

    @property
    def parse_example_spec(self):
        config = {}
        for categorical_column in self.categorical_columns:
            config.update(categorical_column.parse_example_spec)

        return config

    @property
    def _parse_example_spec(self):
        return self.parse_example_spec

    def get_config(self):
        from tensorflow.python.feature_column.serialization import (
            serialize_feature_column,
        )  # pylint: disable=g-import-not-at-top

        config = dict(zip(self._fields, self))
        config["categorical_columns"] = tuple(
            [serialize_feature_column(fc) for fc in self.categorical_columns]
        )

        return config

    @classmethod
    def from_config(cls, config, custom_objects=None, columns_by_name=None):
        """See 'FeatureColumn` base class."""
        from tensorflow.python.feature_column.serialization import (
            deserialize_feature_column,
        )  # pylint: disable=g-import-not-at-top

        fc_lib._check_config_keys(config, cls._fields)
        kwargs = fc_lib._standardize_and_copy_config(config)
        kwargs["categorical_columns"] = tuple(
            [
                deserialize_feature_column(c, custom_objects, columns_by_name)
                for c in config["categorical_columns"]
            ]
        )

        return cls(**kwargs)
