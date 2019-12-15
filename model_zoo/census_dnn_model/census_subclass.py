import tensorflow as tf
from tensorflow.python.keras.metrics import accuracy

from model_zoo.census_dnn_model.census_feature_columns import (
    get_feature_columns,
)


class CustomModel(tf.keras.Model):
    def __init__(self, feature_columns):
        super(CustomModel, self).__init__(name="custom_model")
        self.dense_features = tf.keras.layers.DenseFeatures(feature_columns)
        self.dense_1 = tf.keras.layers.Dense(16, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(16, activation="relu")
        self.dense_3 = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        x = self.dense_features(inputs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)

        return x


def custom_model():
    feature_columns = get_feature_columns()
    return CustomModel(feature_columns=feature_columns)


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
