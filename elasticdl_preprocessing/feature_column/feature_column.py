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

import collections
import itertools

import tensorflow as tf
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.feature_column import feature_column_v2 as fc_lib


def concatenated_categorical_column(categorical_columns):
    """A `CategoricalColumn` to concatenate multiple categorical columns.

    Use this when you have many categorical columns and want to map all
    to embedding. It's recommended to concatenate the sparse id tensors
    from the categorical columns above to one sparse id tensor using this
    API, and then map the combined SparseTensor to Embedding. Because
    there is much overhead to create embedding variables for each
    categorical column, the model size will be huge. Using this way, we
    can reduce the model size.

    The output id range of source categorical columns are [0, num_buckets_0),
    [0, num_buckets_1) ... [0, num_buckets_n). The ids will meet conflict
    in the combined sparse id tensor because they all start from 0. In this
    api, we will add offsets in sparse id tensors from each categorical column
    to avoid this conflict. The id range of the concatenated column is
    [0, num_bucket_0 + num_bucket_1 + ... + num_bucket_n).

    Example

    ```python
    id = categorical_column_with_identity("id", num_buckets=32)
    work_class = categorical_column_with_vocabulary_list(
        "work_class",
        vocabulary_list=[
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            "Never-worked",
        ],
    )
    concat = concatenated_categorical_column([id, work_class])
    ```

    For the feature inputs:
    {
        "id": tf.constant([[1], [-1], [8]]),
        "work_class": tf.constant(
            [
                [""],
                ["Private"],
                ["Self-emp-inc"]
            ], tf.string)
    }

    The sparse id tensor from `id` column is:
    shape = [3,1]
    [0,0]: 1
    [2,0]: 8

    The sparse id tensor from `workclass` column is:
    shape = [3,1]
    [1,0]: 0
    [2,0]: 2

    The concatenated sparse id tensor from `concat` column is:
    shape = [3,2]
    [0,0]: 1
    [1,0]: 32
    [2,0]: 8
    [2,1]: 34

    Returns:
        A `CategoricalColumn` to concatenate multiple categorical columns.

    Raises:
        ValueError: `categorical_columns` is missing or not a list.
        ValueError: `categorical_columns` contains any element which
            is not CategoricalColumn
    """
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

    return ConcatenatedCategoricalColumn(
        categorical_columns=tuple(categorical_columns)
    )


class ConcatenatedCategoricalColumn(
    fc_lib.CategoricalColumn,
    fc_old._CategoricalColumn,
    collections.namedtuple(
        "ConcatenatedCategoricalColumn", ("categorical_columns")
    ),
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
