import os

import tensorflow as tf

from elasticdl.python.common.constants import ODPSConfig
from elasticdl.python.common.odps_io import ODPSWriter
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


def loss(output, labels):
    labels = tf.reshape(labels, [-1])
    return tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output, labels=labels
        )
    )


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def dataset_fn(dataset, training=True):
    def _parse_data(record):
        feature_description = {
            "image": tf.io.FixedLenFeature([32, 32, 3], tf.float32),
            "label": tf.io.FixedLenFeature([1], tf.int64),
        }
        r = tf.io.parse_single_example(record, feature_description)
        label = r["label"]
        image = r["image"]
        return image, label

    dataset = dataset.map(_parse_data)
    dataset = dataset.map(
        lambda x, y: (
            {"image": tf.math.divide(tf.cast(x, tf.float32), 255.0)},
            tf.cast(y, tf.int32),
        )
    )
    if training:
        dataset = dataset.shuffle(buffer_size=1024)
    return dataset


def eval_metrics_fn(predictions, labels):
    labels = tf.reshape(labels, [-1])
    return {
        "accuracy": tf.reduce_mean(
            input_tensor=tf.cast(
                tf.equal(
                    tf.argmax(predictions, 1, output_type=tf.dtypes.int32),
                    labels,
                ),
                tf.float32,
            )
        )
    }


class PredictionOutputsProcessor(BasePredictionOutputsProcessor):
    def __init__(self):
        if all(
            k in os.environ
            for k in (
                os.environ[ODPSConfig.PROJECT_NAME],
                os.environ[ODPSConfig.ACCESS_ID],
                os.environ[ODPSConfig.ACCESS_KEY],
                os.environ[ODPSConfig.ENDPOINT],
            )
        ):
            self.odps_writer = ODPSWriter(
                os.environ[ODPSConfig.PROJECT_NAME],
                os.environ[ODPSConfig.ACCESS_ID],
                os.environ[ODPSConfig.ACCESS_KEY],
                os.environ[ODPSConfig.ENDPOINT],
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
            print(predictions.numpy())
