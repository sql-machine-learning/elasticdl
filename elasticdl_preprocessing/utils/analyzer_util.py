"""This util gets statistics from the environment and return a default value
if there is no statistics in the environment. So user can set a default value
to perform unit tests.
"""
import os

import tensorflow as tf


def get_min(feature_name, default_value):
    """Get the min value of numeric feature from the environment.
    Return the default value if there is no the statistics in
    the environment.

    Args:
        feature_name: String, feature name or column name in a table
        default_value: Float.

    Return:
        Float
    """
    env_name = feature_name + "_min"
    min_value = os.getenv(env_name, None)
    if min_value is None:
        return default_value
    else:
        return float(min_value)


def get_max(feature_name, default_value):
    """Get the max value of numeric feature from the environment.
    Return the default value if there is no the statistics in
    the environment.

    Args:
        feature_name: String, feature name or column name in a table
        default_value: Float.

    Return:
        Float
    """
    env_name = feature_name + "_max"
    max_value = os.getenv(env_name, None)
    if max_value is None:
        return default_value
    else:
        return float(max_value)


def get_avg(feature_name, default_value):
    """Get the average of numeric feature from the environment.
    Return the default value if there is no the statistics in
    the environment.

    Args:
        feature_name: String, feature name or column name in a table
        default_value: Float.

    Return:
        Float
    """
    env_name = feature_name + "_avg"
    mean = os.getenv(env_name, None)
    if mean is None:
        return default_value
    else:
        return float(mean)


def get_stddev(feature_name, default_value):
    """Get the standard deviation from the environment.
    Return the default value if there is no the statistics in
    the environment.

    Args:
        feature_name: String, feature name or column name in a table
        default_value: Float.

    Return:
        Float.
    """
    env_name = feature_name + "_stddev"
    std_dev = os.getenv(env_name, None)
    if std_dev is None:
        return default_value
    else:
        return float(std_dev)


def get_bucket_boundaries(feature_name, default_value):
    """Get the bucket boundaries from the environment.
    Return the default value if there is no the statistics in
    the environment.

    Args:
        feature_name: String, feature name or column name in a table
        default_value: List with float values.

    Return:
        List with float values.
    """
    env_name = feature_name + "_bkt_boundaries"
    boundaries = os.getenv(env_name, None)
    if boundaries is None:
        return default_value
    else:
        boundaries = list(map(float, boundaries.split(",")))
        return sorted(set(boundaries))


def get_value_count(feature_name, default_value):
    """Get the count of feature values set from the environment.
    Return the default value if there is no the statistics in
    the environment.

    Args:
        feature_name: String, feature name or column name in a table
        default_value: Integer.

    Return:
        Integer.
    """
    env_name = feature_name + "_count"
    count = os.getenv(env_name, None)
    if count is None:
        return default_value
    else:
        return int(count)


def get_vocabulary(feature_name, default_value):
    """Get the feature vocabulary from the environment.
    Return the default value if there is no the statistics in
    the environment.

    Args:
        feature_name: String, feature name or column name in a table
        default_value: List with strings.

    Return:
        List with strings.
    """
    env_name = feature_name + "_vocab_path"
    vocabulary_path = os.getenv(env_name, None)
    if vocabulary_path is None:
        return default_value
    else:
        return _read_vocabulary_from_file(vocabulary_path)


def _read_vocabulary_from_file(vocabulary_path):
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
            if token:
                vocab.append(token)
    return vocab
