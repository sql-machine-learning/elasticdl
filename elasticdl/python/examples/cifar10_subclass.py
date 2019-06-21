import tensorflow as tf
import numpy as np


class Cifar10Model(tf.keras.Model):
    def __init__(self, channel_last=True):
        super(Cifar10Model, self).__init__(name="cifar10_model")

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
        x = self._conv_1(inputs)
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


model = Cifar10Model()


def loss(output, labels):
    return tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output, labels=labels.flatten()
        )
    )


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def input_fn(record_list, decode_fn):
    image_numpy_list = []
    label_list = []
    # deserialize
    for r in record_list:
        example = decode_fn(r)

        image_array = example.features.feature['image'].float_list.value
        image_numpy = np.asarray(image_array).reshape(32, 32, 3)
        image_numpy_list.append(image_numpy.astype(np.float32) / 255)

        label = example.features.feature['label'].int64_list.value[0]
        label_list.append(label)

    # batching
    batch_size = len(image_numpy_list)
    images = np.concatenate(image_numpy_list, axis=0)
    images = np.reshape(images, (batch_size, 32, 32, 3))
    image_tensor = tf.convert_to_tensor(value=images)
    label_nparray = np.array(label_list)
    return ([image_tensor], label_nparray)


def eval_metrics_fn(predictions, labels):
    return {
        "accuracy": tf.reduce_mean(
            input_tensor=tf.cast(
                tf.equal(
                    tf.argmax(input=predictions, axis=1), labels.flatten()
                ),
                tf.float32,
            )
        )
    }
