import os

import tensorflow as tf

from elasticdl.python.common.constants import MaxComputeConfig, Mode
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.data.odps_io import ODPSWriter, is_odps_configured
from elasticdl.python.elasticdl.callbacks import LearningRateScheduler
from elasticdl.python.worker.prediction_outputs_processor import (
    BasePredictionOutputsProcessor,
)


def custom_model():
    inputs = tf.keras.layers.Input(shape=(32, 32, 3), name="image")
    use_bias = True

    conv = tf.keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
        use_bias=use_bias,
        activation=None,
    )(inputs)
    bn = tf.keras.layers.BatchNormalization(
        epsilon=1e-06, axis=-1, momentum=0.9
    )(conv)
    activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

    conv = tf.keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
        use_bias=use_bias,
        activation=None,
    )(activation)
    bn = tf.keras.layers.BatchNormalization(
        epsilon=1e-06, axis=-1, momentum=0.9
    )(conv)
    activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

    max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(activation)
    dropout = tf.keras.layers.Dropout(0.2)(max_pool)

    conv = tf.keras.layers.Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        use_bias=use_bias,
        activation=None,
    )(dropout)
    bn = tf.keras.layers.BatchNormalization(
        epsilon=1e-06, axis=-1, momentum=0.9
    )(conv)
    activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

    conv = tf.keras.layers.Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        use_bias=use_bias,
        activation=None,
    )(activation)
    bn = tf.keras.layers.BatchNormalization(
        epsilon=1e-06, axis=-1, momentum=0.9
    )(conv)
    activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

    max_pool = tf.keras.layers.MaxPooling2D()(activation)
    dropout = tf.keras.layers.Dropout(0.3)(max_pool)

    conv = tf.keras.layers.Conv2D(
        128,
        kernel_size=(3, 3),
        padding="same",
        use_bias=use_bias,
        activation=None,
    )(dropout)
    bn = tf.keras.layers.BatchNormalization(
        epsilon=1e-06, axis=-1, momentum=0.9
    )(conv)
    activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

    conv = tf.keras.layers.Conv2D(
        128,
        kernel_size=(3, 3),
        padding="same",
        use_bias=use_bias,
        activation=None,
    )(activation)
    bn = tf.keras.layers.BatchNormalization(
        epsilon=1e-06, axis=-1, momentum=0.9
    )(conv)
    activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

    max_pool = tf.keras.layers.MaxPooling2D()(activation)
    dropout = tf.keras.layers.Dropout(0.4)(max_pool)

    flatten = tf.keras.layers.Flatten()(dropout)
    outputs = tf.keras.layers.Dense(10, name="output")(flatten)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="cifar10_model")


def loss(labels, predictions):
    labels = tf.reshape(labels, [-1])
    return tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=predictions, labels=labels
        )
    )


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def callbacks():
    def _schedule(model_version):
        if model_version < 5000:
            return 0.1
        elif model_version < 15000:
            return 0.01
        else:
            return 0.001

    LearningRateScheduler(_schedule)


def dataset_fn(dataset, mode, _):
    def _parse_data(record):
        if mode == Mode.PREDICTION:
            feature_description = {
                "image": tf.io.FixedLenFeature([32, 32, 3], tf.float32)
            }
        else:
            feature_description = {
                "image": tf.io.FixedLenFeature([32, 32, 3], tf.float32),
                "label": tf.io.FixedLenFeature([1], tf.int64),
            }
        r = tf.io.parse_single_example(record, feature_description)
        features = {
            "image": tf.math.divide(tf.cast(r["image"], tf.float32), 255.0)
        }
        if mode == Mode.PREDICTION:
            return features
        else:
            return features, tf.cast(r["label"], tf.int32)

    dataset = dataset.map(_parse_data)

    if mode == Mode.TRAINING:
        dataset = dataset.shuffle(buffer_size=1024)
    return dataset


def eval_metrics_fn():
    return {
        "accuracy": lambda labels, predictions: tf.equal(
            tf.argmax(predictions, 1, output_type=tf.int32),
            tf.cast(tf.reshape(labels, [-1]), tf.int32),
        )
    }


class PredictionOutputsProcessor(BasePredictionOutputsProcessor):
    def __init__(self):
        if is_odps_configured():
            self.odps_writer = ODPSWriter(
                os.environ[MaxComputeConfig.PROJECT_NAME],
                os.environ[MaxComputeConfig.ACCESS_ID],
                os.environ[MaxComputeConfig.ACCESS_KEY],
                os.environ[MaxComputeConfig.ENDPOINT],
                "cifar10_prediction_outputs",
                # TODO: Print out helpful error message if the columns and
                # column_types do not match with the prediction outputs
                columns=["f" + str(i) for i in range(10)],
                column_types=["double" for _ in range(10)],
            )
        else:
            self.odps_writer = None

    def process(self, predictions, worker_id):
        if self.odps_writer:
            self.odps_writer.from_iterator(
                iter(predictions.numpy().tolist()), worker_id
            )
        else:
            logger.info(predictions.numpy())
