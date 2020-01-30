import tensorflow as tf

from model_zoo.census_wide_deep_model.feature_config import (
    CATEGORICAL_FEATURE_KEYS,
    NUMERIC_FEATURE_KEYS,
    LABEL_KEY,
    FEATURE_GROUPS,
    MODEL_INPUTS,
    TransformOp,
)
from model_zoo.census_wide_deep_model.feature_info_util import (
    get_id_boundaries,
)
from model_zoo.census_wide_deep_model.keras_process_layer import (
    AddIdOffset,
    CategoryHash,
    CategoryLookup,
    NumericBucket,
)


def custom_model():
    # The codes in the method can all be auto-generated
    input_layers = get_input_layers(FEATURE_GROUPS)
    transform_results = transform(input_layers, FEATURE_GROUPS)

    params = {}
    params["id_group_dims"] = get_id_group_dims(FEATURE_GROUPS)
    params["embedding_config"] = MODEL_INPUTS
    model = wide_deep_model(input_layers, transform_results, params)
    return model


def loss(labels, predictions):
    logits = predictions["logits"]
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(tf.reshape(labels, (-1, 1)), tf.float32),
            logits=logits,
        )
    )


def optimizer(lr=0.001):
    return tf.keras.optimizers.Adam(learning_rate=lr)


def eval_metrics_fn():
    return {
        "logits": {
            "accuracy": lambda labels, predictions: tf.equal(
                tf.cast(tf.reshape(predictions, [-1]) > 0.5, tf.int32),
                tf.cast(tf.reshape(labels, [-1]), tf.int32),
            )
        },
        "probs": {"auc": tf.keras.metrics.AUC()},
    }


def learning_rate_scheduler(model_version):
    if model_version < 5000:
        return 0.0003
    elif model_version < 12000:
        return 0.0002
    else:
        return 0.0001


def dataset_fn(dataset, mode, _):
    def _parse_data(record):
        feature_description = dict(
            [
                (name, tf.io.FixedLenFeature((1,), tf.string))
                for name in CATEGORICAL_FEATURE_KEYS
            ]
            + [
                (name, tf.io.FixedLenFeature((1,), tf.float32))
                for name in NUMERIC_FEATURE_KEYS
            ]
            + [(LABEL_KEY, tf.io.FixedLenFeature([], tf.int64))]
        )

        parsed_record = tf.io.parse_single_example(record, feature_description)
        label = parsed_record.pop(LABEL_KEY)

        return parsed_record, label

    dataset = dataset.map(_parse_data)

    return dataset


def get_input_layers(feature_groups):
    input_layers = {}
    for feature_group in feature_groups.values():
        for feature_info in feature_group:
            input_layers[feature_info.name] = tf.keras.layers.Input(
                name=feature_info.name, shape=(1,), dtype=feature_info.dtype
            )
    return input_layers


def transform(inputs, feature_groups):
    result = {}
    for group_name, feature_group in feature_groups.items():
        result[group_name] = transform_group(inputs, feature_group)
    return result


def transform_group(inputs, feature_group):
    """Transform the inputs and concatenate inputs in a group
    to a dense tensor
    """
    group_items = []
    for feature_info in feature_group:
        layer = get_transform_layer(feature_info)
        transform_output = layer(inputs[feature_info.name])
        group_items.append(transform_output)

    id_offsets = get_id_boundaries(feature_group)

    if id_offsets is not None:
        group_items = AddIdOffset(id_offsets[0:-1])(group_items)
    group_stack = tf.keras.layers.concatenate(group_items, axis=-1)
    return group_stack


def get_transform_layer(feature_info):
    if feature_info.op_name == TransformOp.LOOKUP:
        return CategoryLookup(feature_info.param)
    elif feature_info.op_name == TransformOp.HASH:
        return CategoryHash(feature_info.param)
    elif feature_info.op_name == TransformOp.BUCKETIZE:
        return NumericBucket(feature_info.param)
    else:
        raise ValueError("The op is not supported")


def get_id_group_dims(feature_groups):
    id_group_dims = {}
    for group_name, group_features in feature_groups.items():
        id_boundaries = get_id_boundaries(group_features)
        id_group_dims[group_name] = id_boundaries[-1]
    return id_group_dims


def wide_deep_model(input_layers, input_tensors, params):
    id_group_dims = params.get("id_group_dims", {})
    embedding_config = params.get("embedding_config", {})
    wide_embedding_groups = embedding_config["wide"]
    deep_embedding_groups = embedding_config["deep"]
    # wide part
    wide_embeddings = []
    for group_name in wide_embedding_groups:
        wide_embedding_layer = tf.keras.layers.Embedding(
            id_group_dims[group_name], 1
        )
        embedding = wide_embedding_layer(input_tensors[group_name])
        embedding_sum = tf.keras.backend.sum(embedding, axis=1)
        wide_embeddings.append(embedding_sum)

    # deep part
    deep_embeddings = []
    for group_name in deep_embedding_groups:
        deep_embedding_layer = tf.keras.layers.Embedding(
            id_group_dims[group_name], 8
        )
        embedding = deep_embedding_layer(input_tensors[group_name])
        embedding_sum = tf.keras.backend.sum(embedding, axis=1)
        deep_embeddings.append(embedding_sum)

    logits, probs = wide_deep(wide_embeddings, deep_embeddings)

    return tf.keras.Model(
        inputs=input_layers,
        outputs={"logits": logits, "probs": probs},
        name="wide_deep",
    )


def wide_deep(wide_embeddings, deep_embeddings):
    # Wide Part
    wide = tf.keras.layers.Concatenate()(wide_embeddings)  # shape = (None, 3)

    # Deep Part
    dnn_input = tf.reshape(deep_embeddings, shape=(-1, 3 * 8))
    for i in [16, 8, 4]:
        dnn_input = tf.keras.layers.Dense(i)(dnn_input)

    # Output Part
    concat_input = tf.concat([wide, dnn_input], 1)

    logits = tf.reduce_sum(concat_input, 1, keepdims=True)
    probs = tf.reshape(tf.sigmoid(logits), shape=(-1,))
    return logits, probs
