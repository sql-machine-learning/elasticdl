import tensorflow as tf

from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl_preprocessing.layers import (
    ConcatenateWithOffset,
    Discretization,
    Hashing,
    Normalizer,
)
from model_zoo.dac_ctr.feature_config import (
    BUCKET_FEATURES,
    FEATURE_BOUNDARIES,
    FEATURE_DISTINCT_COUNT,
    FEATURES_AVGS,
    FEATURES_STDDEVS,
    HASH_FEATURES,
    STANDARDIZED_FEATURES,
)

MAX_HASHING_BUCKET_SIZE = 1000000


def transform_feature(inputs, feature_groups):
    """According to the FeatureConfig object and feature groups to
    transform inputs to dense tensors.

    Args:
        inputs: A dict contains Keras inputs where the key is the
            feature name and the value is the Keras input.
        feature_groups: 2-D list. each sub-list contains feature names
            of a group

    Returns:
        standardized_tensor: A float tensor
        group_tensors: A dict where the key is group name like "group_0"
            and the value is the integer tensor.
        group_max_ids: A dict which has the same keys as group_tensors and
            the value is the max value of the integer tensor in group_tensors.
    """
    standardized_outputs = []
    for feature in STANDARDIZED_FEATURES:
        standardized_result = Normalizer(
            subtractor=FEATURES_AVGS[feature],
            divisor=FEATURES_STDDEVS[feature],
        )(inputs[feature])
        standardized_outputs.append(standardized_result)

    numerical_tensor = (
        tf.concat(standardized_outputs, -1) if standardized_outputs else None
    )

    if not feature_groups:
        feature_names = BUCKET_FEATURES + HASH_FEATURES
        feature_groups = [feature_names]

    id_tensors = {}
    max_ids = {}
    for i, features in enumerate(feature_groups):
        group_name = "group_{}".format(i)
        id_tensor, max_id = transform_group(inputs, features)
        id_tensors[group_name] = id_tensor
        max_ids[group_name] = max_id

    return numerical_tensor, id_tensors, max_ids


def transform_group(inputs, features):
    """Transform the inputs and concatenate inputs in a group
    to a dense tensor

    Args:
        inputs: A dict contains Keras inputs where the key is the
            feature name and the value is the Keras input.
        features: A list which contains feature names.

    Returns:
        A integer tensor,
        max_id: The max value of the returned tensor
    """
    group_items = []
    id_offsets = [0]
    for feature in features:
        if feature in BUCKET_FEATURES:
            discretize_layer = Discretization(bins=FEATURE_BOUNDARIES[feature])
            transform_output = discretize_layer(inputs[feature])
            group_items.append(transform_output)
            id_offsets.append(id_offsets[-1] + len(discretize_layer.bins) + 1)
            logger.info("{}:{}".format(feature, discretize_layer.bins))
        elif feature in HASH_FEATURES:
            num_bins = FEATURE_DISTINCT_COUNT[feature]
            if num_bins > MAX_HASHING_BUCKET_SIZE:
                num_bins = MAX_HASHING_BUCKET_SIZE
            hash_layer = Hashing(num_bins=num_bins)
            transform_output = hash_layer(inputs[feature])
            id_offsets.append(id_offsets[-1] + hash_layer.num_bins)
            group_items.append(transform_output)
            logger.info("{}:{}".format(feature, hash_layer.num_bins))
        else:
            logger.warning(
                "The preprocessing is not configured for the feature "
                "{}".format(feature)
            )
    concated = ConcatenateWithOffset(id_offsets[0:-1])(group_items)
    max_id = id_offsets[-1]
    return concated, max_id
