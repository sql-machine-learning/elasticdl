import itertools

import tensorflow as tf

from model_zoo.census_model_sqlflow.wide_and_deep.feature_configs import (
    FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY,
    INPUT_SCHEMAS,
    TRANSFORM_OUTPUTS,
    age_bucketize,
    capital_gain_bucketize,
    capital_loss_bucketize,
    education_hash,
    group1,
    group1_embedding_deep,
    group1_embedding_wide,
    group2,
    group2_embedding_deep,
    group2_embedding_wide,
    group3,
    group3_embedding_deep,
    hours_per_week_bucketize,
    marital_status_lookup,
    native_country_hash,
    occupation_hash,
    race_lookup,
    relationship_lookup,
    sex_lookup,
    workclass_lookup,
)
from model_zoo.census_model_sqlflow.wide_and_deep.feature_info_utils import (
    TransformOp,
)
from model_zoo.census_model_sqlflow.wide_and_deep.keras_process_layers import (
    CategoryHash,
    CategoryLookup,
    Group,
    NumericBucket,
)
from tensorflow import feature_column as fc


# The model definition from model zoo. It's functional style.
# Input Params:
#   input_layers: The input layers dict of feature inputs
#   wide_embeddings: The embedding list for the wide part
#   deep_embeddings: The embedding list for the deep part
def wide_and_deep_classifier(input_layers, wide_embeddings, deep_embeddings):
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
            offsets = list(
                itertools.accumulate([0] + feature_transform_info.param[:-1])
            )
            transformed[feature_transform_info.output_name] = Group(offsets)(
                group_inputs
            )
        elif feature_transform_info.op_name == TransformOp.EMBEDDING:
            # The num_buckets should be calcualte from the group items
            group_identity = fc.categorical_column_with_identity(
                feature_transform_info.input_name,
                num_buckets=feature_transform_info.param[0],
            )
            group_embedding = fc.embedding_column(
                group_identity, dimension=feature_transform_info.param[1]
            )
            transformed[
                feature_transform_info.output_name
            ] = tf.keras.layers.DenseFeatures([group_embedding])(
                {
                    feature_transform_info.input_name: transformed[
                        feature_transform_info.input_name
                    ]
                }
            )
        elif feature_transform_info.op_name == TransformOp.ARRAY:
            transformed[feature_transform_info.output_name] = [
                transformed[name] for name in feature_transform_info.input_name
            ]

    return tuple([transformed[name] for name in TRANSFORM_OUTPUTS])


# The following code has the same logic with the `transform` function above.
# It can be generated from the parsed meta in feature_configs using code_gen.
def transform_from_code_gen(source_inputs):
    inputs = source_inputs.copy()

    education_hash_out = CategoryHash(education_hash.param)(
        inputs["education"]
    )
    occupation_hash_out = CategoryHash(occupation_hash.param)(
        inputs["occupation"]
    )
    native_country_hash_out = CategoryHash(native_country_hash.param)(
        inputs["native_country"]
    )
    workclass_lookup_out = CategoryLookup(workclass_lookup.param)(
        inputs["workclass"]
    )
    marital_status_lookup_out = CategoryLookup(marital_status_lookup.param)(
        inputs["marital_status"]
    )
    relationship_lookup_out = CategoryLookup(relationship_lookup.param)(
        inputs["relationship"]
    )
    race_lookup_out = CategoryLookup(race_lookup.param)(inputs["race"])
    sex_lookup_out = CategoryLookup(sex_lookup.param)(inputs["sex"])
    age_bucketize_out = NumericBucket(age_bucketize.param)(inputs["age"])
    capital_gain_bucketize_out = NumericBucket(capital_gain_bucketize.param)(
        inputs["capital_gain"]
    )
    capital_loss_bucketize_out = NumericBucket(capital_loss_bucketize.param)(
        inputs["capital_loss"]
    )
    hours_per_week_bucketize_out = NumericBucket(
        hours_per_week_bucketize.param
    )(inputs["hours_per_week"])

    group1_out = Group(group1.param)(
        [
            workclass_lookup_out,
            hours_per_week_bucketize_out,
            capital_gain_bucketize_out,
            capital_loss_bucketize_out,
        ]
    )
    group2_out = Group(group2.param)(
        [
            education_hash_out,
            marital_status_lookup_out,
            relationship_lookup_out,
            occupation_hash_out,
        ]
    )
    group3_out = Group(group3.param)(
        [
            age_bucketize_out,
            sex_lookup_out,
            race_lookup_out,
            native_country_hash_out,
        ]
    )

    group1_wide_embedding_column = fc.embedding_column(
        fc.categorical_column_with_identity(
            "group1", num_buckets=group1_embedding_wide.param[0]
        ),
        dimension=group1_embedding_wide.param[1],
    )
    group1_embedding_wide_out = tf.keras.layers.DenseFeatures(
        [group1_wide_embedding_column]
    )({"group1": group1_out})

    group2_wide_embedding_column = fc.embedding_column(
        fc.categorical_column_with_identity(
            "group2", num_buckets=group2_embedding_wide.param[0]
        ),
        dimension=group2_embedding_wide.param[1],
    )
    group2_embedding_wide_out = tf.keras.layers.DenseFeatures(
        [group2_wide_embedding_column]
    )({"group2": group2_out})

    group1_deep_embedding_column = fc.embedding_column(
        fc.categorical_column_with_identity(
            "group1", num_buckets=group1_embedding_deep.param[0]
        ),
        dimension=group1_embedding_deep.param[1],
    )
    group1_embedding_deep_out = tf.keras.layers.DenseFeatures(
        [group1_deep_embedding_column]
    )({"group1": group1_out})

    group2_deep_embedding_column = fc.embedding_column(
        fc.categorical_column_with_identity(
            "group2", num_buckets=group2_embedding_deep.param[0]
        ),
        dimension=group2_embedding_deep.param[1],
    )
    group2_embedding_deep_out = tf.keras.layers.DenseFeatures(
        [group2_deep_embedding_column]
    )({"group2": group2_out})

    group3_deep_embedding_column = fc.embedding_column(
        fc.categorical_column_with_identity(
            "group3", num_buckets=group3_embedding_deep.param[0]
        ),
        dimension=group3_embedding_deep.param[1],
    )
    group3_embedding_deep_out = tf.keras.layers.DenseFeatures(
        [group3_deep_embedding_column]
    )({"group3": group3_out})

    wide_embeddings_out = [group1_embedding_wide_out, group2_embedding_wide_out]
    deep_embeddings_out = [
        group1_embedding_deep_out,
        group2_embedding_deep_out,
        group3_embedding_deep_out,
    ]

    return wide_embeddings_out, deep_embeddings_out


# The entry point of the submitter program
def custom_model():
    input_layers = get_input_layers(input_schemas=INPUT_SCHEMAS)
    wide_embeddings, deep_embeddings = transform(input_layers)

    return wide_and_deep_classifier(
        input_layers, wide_embeddings, deep_embeddings
    )


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
