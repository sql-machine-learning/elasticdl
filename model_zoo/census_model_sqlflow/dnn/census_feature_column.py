import tensorflow as tf

CATEGORICAL_FEATURE_KEYS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
NUMERIC_FEATURE_KEYS = [
    "age",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
LABEL_KEY = "label"


def get_feature_columns():
    feature_columns = []

    for numeric_feature_key in NUMERIC_FEATURE_KEYS:
        numeric_feature = tf.feature_column.numeric_column(numeric_feature_key)
        feature_columns.append(numeric_feature)

    for categorical_feature_key in CATEGORICAL_FEATURE_KEYS:
        embedding_feature = tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                categorical_feature_key, hash_bucket_size=64
            ),
            dimension=16,
        )
        feature_columns.append(embedding_feature)

    return feature_columns


def get_feature_input_layers():
    feature_input_layers = {}

    for numeric_feature_key in NUMERIC_FEATURE_KEYS:
        feature_input_layers[numeric_feature_key] = tf.keras.Input(
            shape=(1,), name=numeric_feature_key, dtype=tf.float32
        )

    for categorical_feature_key in CATEGORICAL_FEATURE_KEYS:
        feature_input_layers[categorical_feature_key] = tf.keras.Input(
            shape=(1,), name=categorical_feature_key, dtype=tf.string
        )

    return feature_input_layers
