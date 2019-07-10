import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend, layers, regularizers

try:
    from resnet50_subclass.resnet50_model import (
        L2_WEIGHT_DECAY,
        BATCH_NORM_DECAY,
        BATCH_NORM_EPSILON,
        IdentityBlock,
        ConvBlock,
    )
except ImportError:
    from elasticdl.python.examples.resnet50_subclass.resnet50_model import (
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

    def call(self, inputs, training=False):

        images = inputs["image"]
        if backend.image_data_format() == "channels_first":
            x = self._lambda(images)
        else:
            x = images
        x = self._padding(x)
        x = self._conv2d_1(x)
        x = self._bn_1(x)
        x = self._activation_1(x)
        x = self._maxpooling2d(x)
        x = self._conv_block_1(x)
        x = self._identity_block_1(x)
        x = self._identity_block_2(x)
        x = self._conv_block_2(x)
        x = self._identity_block_3(x)
        x = self._identity_block_4(x)
        x = self._identity_block_5(x)
        x = self._conv_block_3(x)
        x = self._identity_block_6(x)
        x = self._identity_block_7(x)
        x = self._identity_block_8(x)
        x = self._identity_block_9(x)
        x = self._identity_block_10(x)
        x = self._conv_block_4(x)
        x = self._identity_block_11(x)
        x = self._identity_block_12(x)
        x = self._lamba_2(x)
        x = self._dense(x)
        x = backend.cast(x, "float32")
        return self._activation_2(x)


def loss(output, labels):
    return tf.reduce_mean(
        input_tensor=tf.keras.losses.sparse_categorical_crossentropy(
            labels.flatten(), output
        )
    )


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def input_fn(records):
    feature_description = {
        "image": tf.io.FixedLenFeature([224, 224, 3], tf.float32),
        "label": tf.io.FixedLenFeature([1], tf.int64),
    }
    image_list = []
    label_list = []
    for r in records:
        # deserialization
        r = tf.io.parse_single_example(r, feature_description)
        label = r["label"].numpy()
        image = r["image"].numpy()
        # image = cv2.resize(image, (224, 224))
        # processing data
        image = image.astype(np.float32)
        image /= 255
        label = label.astype(np.int32)
        image_list.append(image)
        label_list.append(label)

    # batching
    batch_size = len(image_list)
    images = np.concatenate(image_list, axis=0)
    if tf.keras.backend.image_data_format() == "channels_first":
        images = np.reshape(images, (batch_size, 3, 224, 224))
    else:
        images = np.reshape(images, (batch_size, 224, 224, 3))
    images = tf.convert_to_tensor(value=images)
    labels = np.array(label_list)
    return ({"image": images}, labels)


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
