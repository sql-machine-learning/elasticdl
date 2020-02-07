import tensorflow as tf
from tensorflow.python.keras.metrics import accuracy

from model_zoo.census_model_sqlflow.dnn.census_feature_column import (
    get_feature_columns,
    get_feature_input_layers,
)


# The model definition from model zoo
# Input Params:
#   feature_columns: The feature column array.
#       It can be generated from `COLUMN` clause.
#   feature_input_layers: The input layers specify the feature inputs.
def dnn_classifier(feature_columns, feature_input_layers):
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    x = feature_layer(feature_input_layers)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    y = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=feature_input_layers, outputs=y)

    return model


# The entry point of the submitter program
def custom_model():
    feature_columns = get_feature_columns()
    feature_input_layers = get_feature_input_layers()

    return dnn_classifier(
        feature_columns=feature_columns,
        feature_input_layers=feature_input_layers,
    )


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


# TODO: The dataset_fn and the column names above is bound with
# the input data source. We can consider move it out of the
# model definition file. Currently ElasticDL framework has the
# limitation that the dataset_fn is in the same file with model def.
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
