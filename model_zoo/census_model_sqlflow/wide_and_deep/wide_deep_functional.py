import itertools
import tensorflow as tf

from model_zoo.census_model_sqlflow.wide_and_deep.feature_configs import (
    FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY,
    INPUT_SCHEMAS,
    TRANSFORM_OUTPUTS,
)
from model_zoo.census_model_sqlflow.wide_and_deep.feature_info_utils import (
    FeatureTransformInfo,
    SchemaInfo,
    TransformOp,
)
from model_zoo.census_model_sqlflow.wide_and_deep.keras_process_layers import (
    CategoryHash,
    CategoryLookup,
    Group,
    NumericBucket,
)

# The model definition from model zoo. It's functional style.
# Input Params:
#   input_layers: The input layers dict of feature inputs
#   wide_embeddings: The embedding list for the wide part
#   deep_embeddings: The embedding list for the deep part
def wide_deep_model(input_layers, wide_embeddings, deep_embeddings):
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

    return tf.keras.Model(
        inputs=input_layers,
        outputs={"logits": logits, "probs": probs},
        name="wide_deep",
    )

# Build the input layers from the schema of the input features
def get_input_layers(input_schemas):
    input_layers = {}

    for schema_info in input_schemas:
        input_layers[schema_info.name] = tf.keras.layers.Input(
                name=schema_info.name, shape=(1,), dtype=schema_info.dtype
            )

    return input_layers


# Build the transform logic from the metadata in feature_configs.py.
def transform(inputs):
    transformed = inputs.copy()

    for feature_transform_info in FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY:
        if feature_transform_info.op_name == TransformOp.HASH:
            transformed[feature_transform_info.output_name] = CategoryHash(
                feature_transform_info.param
            )(transformed[feature_transform_info.input_name])
        elif feature_transform_info.op_name == TransformOp.BUCKETIZE:
            transformed[feature_transform_info.output_name] = NumericBucket(
                feature_transform_info.param
            )(transformed[feature_transform_info.input_name])
        elif feature_transform_info.op_name == TransformOp.LOOKUP:
            transformed[feature_transform_info.output_name] = CategoryLookup(
                feature_transform_info.param
            )(transformed[feature_transform_info.input_name])
        elif feature_transform_info.op_name == TransformOp.GROUP:
            group_inputs = [
                transformed[name] for name in feature_transform_info.input_name
            ]
            offsets = list(itertools.accumulate([0] + feature_transform_info.param[:-1]))
            transformed[feature_transform_info.output_name] = Group(None)(
                group_inputs
            )
        elif feature_transform_info.op_name == TransformOp.EMBEDDING:
            # The num_buckets should be calcualte from the group items
            group_identity = tf.feature_column.categorical_column_with_identity(
                feature_transform_info.input_name,
                num_buckets=feature_transform_info.param[0]
            )
            group_embedding = tf.feature_column.embedding_column(
                group_identity,
                dimension=feature_transform_info.param[1]
            )
            transformed[feature_transform_info.output_name] = tf.keras.layers.DenseFeatures(
                [group_embedding]
            )({
                feature_transform_info.input_name: transformed[feature_transform_info.input_name]
            })
        elif feature_transform_info.op_name == TransformOp.ARRAY:
            transformed[feature_transform_info.output_name] = [
                transformed[name] for name in feature_transform_info.input_name
            ]

    return tuple([transformed[name] for name in TRANSFORM_OUTPUTS])


# The entry point of the submitter program
def custom_model():
    input_layers = get_input_layers(input_schemas=INPUT_SCHEMAS)
    wide_embeddings, deep_embeddings = transform(
        input_layers
    )

    return wide_deep_model(input_layers, wide_embeddings, deep_embeddings)


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
