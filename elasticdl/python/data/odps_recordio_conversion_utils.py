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

import os
from collections import namedtuple

import numpy as np
import recordio
import tensorflow as tf


def _find_features_indices(
    features_list, int_features, float_features, bytes_features
):
    """Finds the indices for different types of features."""
    FeatureIndices = namedtuple(
        "FeatureIndices", ["int_features", "float_features", "bytes_features"]
    )
    int_features_indices = [features_list.index(key) for key in int_features]
    float_features_indices = [
        features_list.index(key) for key in float_features
    ]
    bytes_features_indices = [
        features_list.index(key) for key in bytes_features
    ]
    return FeatureIndices(
        int_features_indices, float_features_indices, bytes_features_indices
    )


def _parse_row_to_example(record, features_list, feature_indices):
    """
    Parses one row (a flat list or one-dimensional numpy array)
    to a TensorFlow Example.
    """
    if isinstance(record, list):
        record = np.array(record, dtype=object)

    example = tf.train.Example()
    # Note: these cannot be constructed dynamically since
    # we cannot assign a value to an embedded message
    # field in protobuf
    for feature_ind in feature_indices.int_features:
        example.features.feature[
            features_list[feature_ind]
        ].int64_list.value.append(
            int(_maybe_encode_unicode_string(record[feature_ind]) or 0)
        )
    for feature_ind in feature_indices.float_features:
        example.features.feature[
            features_list[feature_ind]
        ].float_list.value.append(
            float(_maybe_encode_unicode_string(record[feature_ind]) or 0.0)
        )
    for feature_ind in feature_indices.bytes_features:
        example.features.feature[
            features_list[feature_ind]
        ].bytes_list.value.append(
            _maybe_encode_unicode_string(record[feature_ind])
        )
    return example


def _maybe_encode_unicode_string(record):
    """Encodes unicode strings if needed."""
    if isinstance(record, str):
        record = bytes(record, "utf-8").strip()
    return record


def _find_feature_indices_from_record(record):
    """Find the indices of different feature types."""
    feature_types = [type(value) for value in record]
    FeatureIndices = namedtuple(
        "FeatureIndices", ["int_features", "float_features", "bytes_features"]
    )
    return FeatureIndices(
        [i for i, x in enumerate(feature_types) if x == int],
        [i for i, x in enumerate(feature_types) if x == float],
        [i for i, x in enumerate(feature_types) if x == str],
    )


def write_recordio_shards_from_iterator(
    records_iter, features_list, output_dir, records_per_shard
):
    """Writes RecordIO files from Python iterator of numpy arrays."""
    # Take the first record batch to check whether it contains multiple items
    first_record_batch = next(records_iter)
    is_first_record_batch_consumed = False
    is_multi_items_per_batch = any(
        isinstance(i, list) for i in first_record_batch
    )

    # Find the features of different types that will be used
    # in `_parse_row_to_example()` later
    record = (
        first_record_batch[0]
        if is_multi_items_per_batch
        else first_record_batch
    )
    feature_indices = _find_feature_indices_from_record(record)

    writer = None
    rows_written = 0
    shards_written = 0
    while True:
        try:
            # Make sure to consume the first record batch
            if is_first_record_batch_consumed:
                record_batch = next(records_iter)
            else:
                record_batch = first_record_batch
                is_first_record_batch_consumed = True
            if not is_multi_items_per_batch:
                record_batch = [record_batch]

            # Write each record in the batch to a RecordIO shard
            for record in record_batch:
                # Initialize the writer for the new shard
                if rows_written % records_per_shard == 0:
                    if writer is not None:
                        writer.close()
                    shard_file_path = os.path.join(
                        output_dir, "data-%05d" % shards_written
                    )
                    writer = recordio.Writer(shard_file_path)
                    shards_written += 1

                writer.write(
                    _parse_row_to_example(
                        record, features_list, feature_indices
                    ).SerializeToString()
                )
                rows_written += 1
        except StopIteration:
            break

    writer.close()
