import tensorflow as tf
from tensorflow.python.keras.metrics import accuracy

from elasticdl.python.elasticdl.feature_column import feature_column

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
OPTIONAL_NUMERIC_FEATURE_KEYS = [
    "education-num",
]
LABEL_KEY = "label"


def get_feature_columns_and_inputs():
    feature_columns = []
    feature_input_layers = {}

    for numeric_feature_key in NUMERIC_FEATURE_KEYS:
        numeric_feature = tf.feature_column.numeric_column(numeric_feature_key)
        feature_columns.append(numeric_feature)
        feature_input_layers[numeric_feature_key] = tf.keras.Input(
            shape=(1,), name=numeric_feature_key, dtype=tf.float32
        )

    for categorical_feature_key in CATEGORICAL_FEATURE_KEYS:
        embedding_feature = feature_column.embedding_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                categorical_feature_key, hash_bucket_size=64
            ),
            dimension=16,
        )
        feature_columns.append(embedding_feature)
        feature_input_layers[categorical_feature_key] = tf.keras.Input(
            shape=(1,), name=categorical_feature_key, dtype=tf.string
        )

    return feature_columns, feature_input_layers


def custom_model():
    feature_columns, feature_inputs = get_feature_columns_and_inputs()
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    x = feature_layer(feature_inputs)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    y = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=feature_inputs, outputs=y)

    return model


def loss(labels, predictions):
    labels = tf.expand_dims(labels, axis=1)
    return tf.keras.losses.binary_crossentropy(labels, predictions)


def optimizer():
    return tf.keras.optimizers.Adam()


def eval_metrics_fn():
    return {
        "accuracy": lambda labels, predictions: accuracy(
            tf.cast(tf.squeeze(tf.round(predictions)), tf.int32),
            tf.cast(labels, tf.int32),
        )
    }


def dataset_fn(dataset, mode, _):
    def _parse_data(record):

        feature_description = dict(
            [
                (name, tf.io.FixedLenFeature([], tf.string))
                for name in CATEGORICAL_FEATURE_KEYS
            ]
            + [
                (name, tf.io.FixedLenFeature([], tf.float32))
                for name in NUMERIC_FEATURE_KEYS
            ]
            + [(LABEL_KEY, tf.io.FixedLenFeature([], tf.int64))]
        )

        parsed_record = tf.io.parse_single_example(record, feature_description)
        label = parsed_record.pop(LABEL_KEY)

        return parsed_record, label

    dataset = dataset.map(_parse_data)

    return dataset
