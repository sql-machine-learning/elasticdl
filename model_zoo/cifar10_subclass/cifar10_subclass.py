import tensorflow as tf

from elasticdl.python.common.constants import Mode
from elasticdl.python.model import ElasticDLKerasModelBase


class CustomModel(ElasticDLKerasModelBase):
    def __init__(self, channel_last=True, context=None):
        super(CustomModel, self).__init__(name="cifar10_model")

        use_bias = True
        self._conv_1 = tf.keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            padding="same",
            use_bias=use_bias,
            activation=None,
        )
        self._bn_1 = tf.keras.layers.BatchNormalization(
            epsilon=1e-06, axis=-1, momentum=0.9
        )
        self._relu_1 = tf.keras.layers.Activation(tf.nn.relu)

        self._conv_2 = tf.keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            padding="same",
            use_bias=use_bias,
            activation=None,
        )
        self._bn_2 = tf.keras.layers.BatchNormalization(
            epsilon=1e-06, axis=-1, momentum=0.9
        )
        self._relu_2 = tf.keras.layers.Activation(tf.nn.relu)

        self._max_pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self._dropout_1 = tf.keras.layers.Dropout(0.2)

        self._conv_3 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
            use_bias=use_bias,
            activation=None,
        )
        self._bn_3 = tf.keras.layers.BatchNormalization(
            epsilon=1e-06, axis=-1, momentum=0.9
        )
        self._relu_3 = tf.keras.layers.Activation(tf.nn.relu)

        self._conv_4 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
            use_bias=use_bias,
            activation=None,
        )
        self._bn_4 = tf.keras.layers.BatchNormalization(
            epsilon=1e-06, axis=-1, momentum=0.9
        )
        self._relu_4 = tf.keras.layers.Activation(tf.nn.relu)

        self._max_pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self._dropout_2 = tf.keras.layers.Dropout(0.3)

        self._conv_5 = tf.keras.layers.Conv2D(
            128,
            kernel_size=(3, 3),
            padding="same",
            use_bias=use_bias,
            activation=None,
        )
        self._bn_5 = tf.keras.layers.BatchNormalization(
            epsilon=1e-06, axis=-1, momentum=0.9
        )
        self._relu_5 = tf.keras.layers.Activation(tf.nn.relu)

        self._conv_6 = tf.keras.layers.Conv2D(
            128,
            kernel_size=(3, 3),
            padding="same",
            use_bias=use_bias,
            activation=None,
        )
        self._bn_6 = tf.keras.layers.BatchNormalization(
            epsilon=1e-06, axis=-1, momentum=0.9
        )
        self._relu_6 = tf.keras.layers.Activation(tf.nn.relu)

        self._max_pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self._dropout_3 = tf.keras.layers.Dropout(0.4)

        self._flatten_1 = tf.keras.layers.Flatten()
        self._dense_1 = tf.keras.layers.Dense(10, name="output")

    def call(self, inputs, training=False):
        x = self._conv_1(inputs["image"])
        x = self._bn_1(x)
        x = self._relu_1(x)
        x = self._conv_2(x)
        x = self._bn_2(x)
        x = self._relu_2(x)
        x = self._max_pool_1(x)
        x = self._dropout_1(x)
        x = self._conv_3(x)
        x = self._bn_3(x)
        x = self._relu_3(x)
        x = self._conv_4(x)
        x = self._bn_4(x)
        x = self._relu_4(x)
        x = self._max_pool_2(x)
        x = self._dropout_2(x)
        x = self._conv_5(x)
        x = self._bn_5(x)
        x = self._relu_5(x)
        x = self._conv_6(x)
        x = self._bn_6(x)
        x = self._relu_6(x)
        x = self._max_pool_3(x)
        x = self._dropout_3(x)
        x = self._flatten_1(x)
        return self._dense_1(x)

    def get_model(self):
        return self

    def loss(self, outputs=None, labels=None):
        labels = tf.reshape(labels, [-1])
        return tf.reduce_mean(
            input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=outputs, labels=labels
            )
        )

    def optimizer(self, lr=0.1):
        return tf.optimizers.SGD(lr)

    def metrics(
        self, mode=Mode.TRAINING, outputs=None, predictions=None, labels=None
    ):
        if mode == Mode.EVALUATION:
            labels = tf.reshape(labels, [-1])
            return {
                "accuracy": tf.reduce_mean(
                    input_tensor=tf.cast(
                        tf.equal(
                            tf.argmax(
                                predictions, 1, output_type=tf.dtypes.int32
                            ),
                            labels,
                        ),
                        tf.float32,
                    )
                )
            }


def dataset_fn(dataset, mode):
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

    if mode != Mode.PREDICTION:
        dataset = dataset.shuffle(buffer_size=1024)
    return dataset
