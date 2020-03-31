import tensorflow as tf
from tensorflow import feature_column as fc

from elasticdl.python.elasticdl.callbacks import LearningRateScheduler
from elasticdl_preprocessing.feature_column import feature_column as edl_fc
from model_zoo.census_model_sqlflow.wide_and_deep.feature_configs import (
    FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY,
    INPUT_SCHEMAS,
    TRANSFORM_OUTPUTS,
    age_bucketize,
    capital_gain_bucketize,
    capital_loss_bucketize,
    education_hash,
    group1_embedding_deep,
    group1_embedding_wide,
    group2_embedding_deep,
    group2_embedding_wide,
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
from model_zoo.census_model_sqlflow.wide_and_deep.transform_ops import (
    TransformOpType,
)


# The model definition from model zoo. It's functional style.
# Input Params:
#   input_layers: The input layers dict of feature inputs
#   wide_embedding: A tensor. Embedding for the wide part.
#   deep_embedding: A tensor. Embedding for the deep part.
def wide_and_deep_classifier(input_layers, wide_embedding, deep_embedding):
    # Wide Part
    wide = wide_embedding  # shape = (None, 3)

    # Deep Part
    dnn_input = deep_embedding
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
    feature_column_dict = {}

    for feature_transform_info in FEATURE_TRANSFORM_INFO_EXECUTE_ARRAY:
        if feature_transform_info.op_type == TransformOpType.HASH:
            feature_column_dict[
                feature_transform_info.output
            ] = tf.feature_column.categorical_column_with_hash_bucket(
                feature_transform_info.input,
                hash_bucket_size=feature_transform_info.hash_bucket_size,
            )
        elif feature_transform_info.op_type == TransformOpType.BUCKETIZE:
            feature_column_dict[
                feature_transform_info.output
            ] = tf.feature_column.bucketized_column(
                fc.numeric_column(feature_transform_info.input),
                boundaries=feature_transform_info.boundaries,
            )
        elif feature_transform_info.op_type == TransformOpType.LOOKUP:
            feature_column_dict[
                feature_transform_info.output
            ] = tf.feature_column.categorical_column_with_vocabulary_list(
                feature_transform_info.input,
                vocabulary_list=workclass_lookup.vocabulary_list,
            )
        elif feature_transform_info.op_type == TransformOpType.CONCAT:
            concat_inputs = [
                feature_column_dict[name]
                for name in feature_transform_info.input
            ]
            concat_column = edl_fc.concatenated_categorical_column(
                concat_inputs
            )
            feature_column_dict[feature_transform_info.output] = concat_column
        elif feature_transform_info.op_type == TransformOpType.EMBEDDING:
            feature_column_dict[
                feature_transform_info.output
            ] = tf.feature_column.embedding_column(
                feature_column_dict[feature_transform_info.input],
                dimension=feature_transform_info.output_dim,
            )
        elif feature_transform_info.op_type == TransformOpType.ARRAY:
            feature_column_dict[feature_transform_info.output] = [
                feature_column_dict[name]
                for name in feature_transform_info.input
            ]

    return tuple(
        [
            tf.keras.layers.DenseFeatures(feature_column_dict[name])(inputs)
            for name in TRANSFORM_OUTPUTS
        ]
    )


# The following code has the same logic with the `transform` function above.
# It can be generated from the parsed meta in feature_configs using code_gen.
def transform_from_code_gen(source_inputs):
    education_hash_fc = fc.categorical_column_with_hash_bucket(
        "education", hash_bucket_size=education_hash.hash_bucket_size
    )

    occupation_hash_fc = fc.categorical_column_with_hash_bucket(
        "occupation", hash_bucket_size=occupation_hash.hash_bucket_size
    )

    native_country_hash_fc = fc.categorical_column_with_hash_bucket(
        "native_country", hash_bucket_size=native_country_hash.hash_bucket_size
    )

    workclass_lookup_fc = fc.categorical_column_with_vocabulary_list(
        "workclass", vocabulary_list=workclass_lookup.vocabulary_list
    )

    marital_status_lookup_fc = fc.categorical_column_with_vocabulary_list(
        "marital_status", vocabulary_list=marital_status_lookup.vocabulary_list
    )

    relationship_lookup_fc = fc.categorical_column_with_vocabulary_list(
        "relationship", vocabulary_list=relationship_lookup.vocabulary_list
    )

    race_lookup_fc = fc.categorical_column_with_vocabulary_list(
        "race", vocabulary_list=race_lookup.vocabulary_list
    )

    sex_lookup_fc = fc.categorical_column_with_vocabulary_list(
        "sex", vocabulary_list=sex_lookup.vocabulary_list
    )

    age_bucketize_fc = fc.bucketized_column(
        fc.numeric_column("age"), boundaries=age_bucketize.boundaries
    )

    capital_gain_bucketize_fc = fc.bucketized_column(
        fc.numeric_column("capital_gain"),
        boundaries=capital_gain_bucketize.boundaries,
    )

    capital_loss_bucketize_fc = fc.bucketized_column(
        fc.numeric_column("capital_loss"),
        boundaries=capital_loss_bucketize.boundaries,
    )

    hours_per_week_bucketize_fc = fc.bucketized_column(
        fc.numeric_column("hours_per_week"),
        boundaries=hours_per_week_bucketize.boundaries,
    )

    group1_fc = edl_fc.concatenated_categorical_column(
        categorical_columns=[
            workclass_lookup_fc,
            hours_per_week_bucketize_fc,
            capital_gain_bucketize_fc,
            capital_loss_bucketize_fc,
        ]
    )

    group2_fc = edl_fc.concatenated_categorical_column(
        categorical_columns=[
            education_hash_fc,
            marital_status_lookup_fc,
            relationship_lookup_fc,
            occupation_hash_fc,
        ]
    )

    group3_fc = edl_fc.concatenated_categorical_column(
        categorical_columns=[
            age_bucketize_fc,
            sex_lookup_fc,
            race_lookup_fc,
            native_country_hash_fc,
        ]
    )

    group1_wide_embedding_fc = fc.embedding_column(
        group1_fc, dimension=group1_embedding_wide.output_dim,
    )

    group2_wide_embedding_fc = fc.embedding_column(
        group2_fc, dimension=group2_embedding_wide.output_dim,
    )

    group1_deep_embedding_fc = fc.embedding_column(
        group1_fc, dimension=group1_embedding_deep.output_dim,
    )

    group2_deep_embedding_fc = fc.embedding_column(
        group2_fc, dimension=group2_embedding_deep.output_dim,
    )

    group3_deep_embedding_fc = fc.embedding_column(
        group3_fc, dimension=group3_embedding_deep.output_dim,
    )

    wide_feature_columns = [
        group1_wide_embedding_fc,
        group2_wide_embedding_fc,
    ]

    deep_feature_columns = [
        group1_deep_embedding_fc,
        group2_deep_embedding_fc,
        group3_deep_embedding_fc,
    ]

    return (
        tf.keras.layers.DenseFeatures(wide_feature_columns)(source_inputs),
        tf.keras.layers.DenseFeatures(deep_feature_columns)(source_inputs),
    )


# The entry point of the submitter program
def custom_model():
    input_layers = get_input_layers(input_schemas=INPUT_SCHEMAS)
    wide_embedding, deep_embedding = transform_from_code_gen(input_layers)

    return wide_and_deep_classifier(
        input_layers, wide_embedding, deep_embedding
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


def callbacks():
    def _schedule(model_version):
        if model_version < 5000:
            return 0.0003
        elif model_version < 12000:
            return 0.0002
        else:
            return 0.0001

    return [LearningRateScheduler(_schedule)]


if __name__ == "__main__":
    model = custom_model()
    print(model.summary())

    output = model.call(
        {
            "education": tf.constant([["Bachelors"]], tf.string),
            "occupation": tf.constant([["Tech-support"]], tf.string),
            "native_country": tf.constant([["United-States"]], tf.string),
            "workclass": tf.constant([["Private"]], tf.string),
            "marital_status": tf.constant([["Separated"]], tf.string),
            "relationship": tf.constant([["Husband"]], tf.string),
            "race": tf.constant([["White"]], tf.string),
            "sex": tf.constant([["Female"]], tf.string),
            "age": tf.constant([[18]], tf.float32),
            "capital_gain": tf.constant([[100.0]], tf.float32),
            "capital_loss": tf.constant([[1.0]], tf.float32),
            "hours_per_week": tf.constant([[40]], tf.float32),
        }
    )

    print(output)
