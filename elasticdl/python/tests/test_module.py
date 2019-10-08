import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input

from elasticdl.python.worker.prediction_outputs_processor import (
    BasePredictionOutputsProcessor,
)


def custom_model():
    inputs = Input(shape=(1, 1), name="x")
    outputs = Dense(1)(inputs)
    return Model(inputs, outputs)


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


def eval_metrics_fn():
    return {"mse": tf.keras.metrics.MeanSquaredError()}


class PredictionOutputsProcessor(BasePredictionOutputsProcessor):
    def __init__(self):
        pass

    def process(self, predictions, worker_id):
        pass
