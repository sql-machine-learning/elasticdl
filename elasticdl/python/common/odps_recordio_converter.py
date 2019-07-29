from collections import namedtuple, OrderedDict

import numpy as np
import tensorflow as tf

from datetime import datetime as _datetime
from odps.types import Tinyint, Smallint, Int, Bigint, Float, Double, String, Datetime, Boolean, Binary


tinyint = Tinyint()
smallint = Smallint()
int_ = Int()
bigint = Bigint()
float_ = Float()
double = Double()
string = String()
datetime = Datetime()
boolean = Boolean()
binary = Binary()

_odps_primitive_data_types = dict(
    [(t.name, t) for t in (
        tinyint, smallint, int_, bigint, float_, double, string, datetime,
        boolean, binary
    )]
)

integer_builtins = (int, np.integer)
float_builtins = (float, np.float)

_odps_primitive_to_builtin_types = OrderedDict((
    (bigint, integer_builtins),
    (tinyint, integer_builtins),
    (smallint, integer_builtins),
    (int_, integer_builtins),
    (double, float_builtins),
    (float_, float_builtins),
    (string, (str, bytes)),
    (binary, bytes),
    (datetime, _datetime),
    (boolean, bool),
))


def infer_primitive_data_type(value):
    for data_type, builtin_types in _odps_primitive_to_builtin_types.items():
        if isinstance(value, builtin_types):
            return data_type


def _find_features_indices(
    features_list, int_features, float_features, bytes_features
):
    """Finds the indices for different types of features."""
    FeatureIndices = namedtuple(
        "FeatureIndices",
        ["int_features", "float_features", "bytes_features"],
        verbose=False,
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
