import tensorflow as tf
from tensorflow.python.keras import backend, layers, regularizers

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


class IdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, stage, block):
        super(IdentityBlock, self).__init__(name="identity_block")
        filters1, filters2, filters3 = filters
        if backend.image_data_format() == "channels_last":
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = "res" + str(stage) + block + "_branch"
        bn_name_base = "bn" + str(stage) + block + "_branch"

        self._conv2d_1 = layers.Conv2D(
            filters1,
            (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            name=conv_name_base + "2a",
        )
        self._bn_1 = layers.BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            name=bn_name_base + "2a",
        )
        self.activation_1 = layers.Activation("relu")

        self._conv2d_2 = layers.Conv2D(
            filters2,
            kernel_size,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            name=conv_name_base + "2b",
        )
        self._bn_2 = layers.BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            name=bn_name_base + "2b",
        )
        self._activation_2 = layers.Activation("relu")

        self._conv2d_3 = layers.Conv2D(
            filters3,
            (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            name=conv_name_base + "2c",
        )
        self._bn_3 = layers.BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            name=bn_name_base + "2c",
        )

        self._activation_e3 = layers.Activation("relu")

    def call(self, inputs, training=False):
        x = self._conv2d_1(inputs)
        x = self._bn_1(x)
        x = self.activation_1(x)
        x = self._conv2d_2(x)
        x = self._bn_2(x)
        x = self._activation_2(x)
        x = self._conv2d_3(x)
        x = self._bn_3(x)
        x = layers.add([x, inputs])
        return self._activation_e3(x)


class ConvBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, stage, block, strides=(2, 2)):
        super(ConvBlock, self).__init__(name="conv_block")
        filters1, filters2, filters3 = filters
        if backend.image_data_format() == "channels_last":
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = "res" + str(stage) + block + "_branch"
        bn_name_base = "bn" + str(stage) + block + "_branch"

        self._conv2d_1 = layers.Conv2D(
            filters1,
            (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            name=conv_name_base + "2a",
        )
        self._bn_1 = layers.BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            name=bn_name_base + "2a",
        )
        self._activation_1 = layers.Activation("relu")

        self._conv2d_2 = layers.Conv2D(
            filters2,
            kernel_size,
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            name=conv_name_base + "2b",
        )
        self._bn_2 = layers.BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            name=bn_name_base + "2b",
        )
        self._activation_2 = layers.Activation("relu")

        self._conv2d_3 = layers.Conv2D(
            filters3,
            (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            name=conv_name_base + "2c",
        )
        self._bn_3 = layers.BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            name=bn_name_base + "2c",
        )

        self._shortcut = layers.Conv2D(
            filters3,
            (1, 1),
            strides=strides,
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            name=conv_name_base + "1",
        )
        self._bn_4 = layers.BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            name=bn_name_base + "1",
        )

        self._activation_4 = layers.Activation("relu")

    def call(self, inputs, training=False):
        x = self._conv2d_1(inputs)
        x = self._bn_1(x)
        x = self._activation_1(x)
        x = self._conv2d_2(x)
        x = self._bn_2(x)
        x = self._activation_2(x)
        x = self._conv2d_3(x)
        x = self._bn_3(x)
        shortcut = self._shortcut(inputs)
        shortcut = self._bn_4(shortcut)
        x = layers.add([x, shortcut])
        return self._activation_4(x)
