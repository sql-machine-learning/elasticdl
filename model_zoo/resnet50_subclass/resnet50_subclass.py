import tensorflow as tf
from tensorflow.python.keras import backend, layers, regularizers

from elasticdl.python.common.constants import Mode

try:
    from resnet50_subclass.resnet50_model import (
        L2_WEIGHT_DECAY,
        BATCH_NORM_DECAY,
        BATCH_NORM_EPSILON,
        IdentityBlock,
        ConvBlock,
    )
except ImportError:
    from model_zoo.resnet50_subclass.resnet50_model import (
        L2_WEIGHT_DECAY,
        BATCH_NORM_DECAY,
        BATCH_NORM_EPSILON,
        IdentityBlock,
        ConvBlock,
    )


class CustomModel(tf.keras.Model):
    def __init__(self, num_classes=10, dtype="float32", batch_size=None):
        super(CustomModel, self).__init__(name="resnet50")

        if backend.image_data_format() == "channels_first":
            self._lambda = layers.Lambda(
                lambda x: backend.permute_dimensions(x, (0, 3, 1, 2)),
                name="transpose",
            )
            bn_axis = 1
            data_format = "channels_first"
        else:
            bn_axis = 3
            data_format = "channels_last"

        self._padding = layers.ZeroPadding2D(
            padding=(3, 3), data_format=data_format, name="zero_pad"
        )
        self._conv2d_1 = layers.Conv2D(
            64,
            (7, 7),
            strides=(2, 2),
            padding="valid",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            name="conv1",
        )
        self._bn_1 = layers.BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            name="bn_conv1",
        )
        self._activation_1 = layers.Activation("relu")
        self._maxpooling2d = layers.MaxPooling2D(
            (3, 3), strides=(2, 2), padding="same"
        )

        self._conv_block_1 = ConvBlock(
            3, [64, 64, 256], stage=2, block="a", strides=(1, 1)
        )
        self._identity_block_1 = IdentityBlock(
            3, [64, 64, 256], stage=2, block="b"
        )
        self._identity_block_2 = IdentityBlock(
            3, [64, 64, 256], stage=2, block="c"
        )

        self._conv_block_2 = ConvBlock(3, [128, 128, 512], stage=3, block="a")
        self._identity_block_3 = IdentityBlock(
            3, [128, 128, 512], stage=3, block="b"
        )
        self._identity_block_4 = IdentityBlock(
            3, [128, 128, 512], stage=3, block="c"
        )
        self._identity_block_5 = IdentityBlock(
            3, [128, 128, 512], stage=3, block="d"
        )

        self._conv_block_3 = ConvBlock(3, [256, 256, 1024], stage=4, block="a")
        self._identity_block_6 = IdentityBlock(
            3, [256, 256, 1024], stage=4, block="b"
        )
        self._identity_block_7 = IdentityBlock(
            3, [256, 256, 1024], stage=4, block="c"
        )
        self._identity_block_8 = IdentityBlock(
            3, [256, 256, 1024], stage=4, block="d"
        )
        self._identity_block_9 = IdentityBlock(
            3, [256, 256, 1024], stage=4, block="e"
        )
        self._identity_block_10 = IdentityBlock(
            3, [256, 256, 1024], stage=4, block="f"
        )

        self._conv_block_4 = ConvBlock(3, [512, 512, 2048], stage=5, block="a")
        self._identity_block_11 = IdentityBlock(
            3, [512, 512, 2048], stage=5, block="b"
        )
        self._identity_block_12 = IdentityBlock(
            3, [512, 512, 2048], stage=5, block="c"
        )

        rm_axes = (
            [1, 2]
            if backend.image_data_format() == "channels_last"
            else [2, 3]
        )
        self._lamba_2 = layers.Lambda(
            lambda x: backend.mean(x, rm_axes), name="reduce_mean"
        )
        self._dense = layers.Dense(
            num_classes,
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            name="fc1000",
        )
        self._activation_2 = layers.Activation("softmax")

    def call(self, inputs, training=True):

        images = inputs["image"]
        if backend.image_data_format() == "channels_first":
            x = self._lambda(images)
        else:
            x = images
        x = self._padding(x)
        x = self._conv2d_1(x)
        x = self._bn_1(x, training=training)
        x = self._activation_1(x)
        x = self._maxpooling2d(x)
        x = self._conv_block_1(x, training=training)
        x = self._identity_block_1(x, training=training)
        x = self._identity_block_2(x, training=training)
        x = self._conv_block_2(x, training=training)
        x = self._identity_block_3(x, training=training)
        x = self._identity_block_4(x, training=training)
        x = self._identity_block_5(x, training=training)
        x = self._conv_block_3(x, training=training)
        x = self._identity_block_6(x, training=training)
        x = self._identity_block_7(x, training=training)
        x = self._identity_block_8(x, training=training)
        x = self._identity_block_9(x, training=training)
        x = self._identity_block_10(x, training=training)
        x = self._conv_block_4(x, training=training)
        x = self._identity_block_11(x, training=training)
        x = self._identity_block_12(x, training=training)
        x = self._lamba_2(x)
        x = self._dense(x)
        x = backend.cast(x, "float32")
        return self._activation_2(x)


def loss(output, labels):
    labels = tf.reshape(labels, [-1])
    return tf.reduce_mean(
        input_tensor=tf.keras.losses.sparse_categorical_crossentropy(
            labels, output
        )
    )


def optimizer(lr=0.02):
    return tf.keras.optimizers.SGD(lr)


def dataset_fn(dataset, mode, _):
    def _parse_data(record):
        if mode == Mode.PREDICTION:
            feature_description = {
                "image": tf.io.FixedLenFeature([], tf.string)
            }
        else:
            feature_description = {
                "image": tf.io.FixedLenFeature([], tf.string),
                "label": tf.io.FixedLenFeature([], tf.int64),
            }
        r = tf.io.parse_single_example(record, feature_description)
        features = tf.image.resize(
            tf.image.decode_jpeg(r["image"]),
            [224, 224],
            method=tf.image.ResizeMethod.BILINEAR,
        )
        features = tf.cond(
            tf.math.greater(tf.size(features), 244 * 244),
            lambda: features,
            lambda: tf.image.grayscale_to_rgb(features),
        )
        features = {
            "image": tf.math.divide(tf.cast(features, tf.float32), 255.0)
        }
        if mode == Mode.PREDICTION:
            return features
        else:
            return features, tf.cast(r["label"] - 1, tf.int32)

    dataset = dataset.map(_parse_data)

    if mode != Mode.PREDICTION:
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
