import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Input

from elasticdl.python.worker.prediction_outputs_processor import (
    BasePredictionOutputsProcessor,
)


def custom_model():
    inputs = Input(shape=(1, 1), name="x")
    outputs = Dense(1)(inputs)
    return Model(inputs, outputs)


def custom_model_with_embedding():
    inputs = Input(shape=(4,), name="x")
    embedding = Embedding(4, 2)(inputs)
    outputs = Dense(1)(embedding)
    return Model(inputs, outputs)


def custom_sequential_model(feature_columns):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.DenseFeatures(feature_columns=feature_columns),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def feature_columns_fn():
    age = tf.feature_column.numeric_column("age", dtype=tf.int64)
    education = tf.feature_column.categorical_column_with_hash_bucket(
        "education", hash_bucket_size=4
    )
    education_one_hot = tf.feature_column.indicator_column(education)
    return [age, education_one_hot]


def loss(predictions, labels):
    return tf.reduce_mean(tf.square(predictions - labels))


def dataset_fn(dataset, mode, metadata):
    def _parse_data(record):
        feature_description = {
            "x": tf.io.FixedLenFeature([1], tf.float32),
            "y": tf.io.FixedLenFeature([1], tf.float32),
        }
        r = tf.io.parse_single_example(record, feature_description)
        return {"x": r["x"]}, r["y"]

    dataset = dataset.map(_parse_data)
    return dataset


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def ftrl_optimizer(lr=0.1):
    return tf.optimizers.Ftrl(lr)


def eval_metrics_fn():
    return {"mse": lambda labels, outputs: tf.square(outputs - labels)}


class PredictionOutputsProcessor(BasePredictionOutputsProcessor):
    def __init__(self):
        pass

    def process(self, predictions, worker_id):
        pass
