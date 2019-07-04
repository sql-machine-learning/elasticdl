import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import utils

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

class ResNet50(tf.keras.Model):
    def __init__(self, num_classes, dtype='float32', batch_size=None):
        super(ResNet50, self).__init__(name="resnet50")

        if backend.image_data_format() == 'channels_first':
            self._lambda = layers.Lambda(lambda x: backend.permute_dimensions(x, (0, 3, 1, 2)),
                              name='transpose')
            bn_axis = 1
        else:
            bn_axis = 3

        self._padding = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')
        self._conv2d_1 = layers.Conv2D(64, (7, 7),
                          strides=(2, 2),
                          padding='valid', use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                          name='conv1')
        self._bn_1 = layers.BatchNormalization(axis=bn_axis,
                                      momentum=BATCH_NORM_DECAY,
                                      epsilon=BATCH_NORM_EPSILON,
                                      name='bn_conv1')
        self._activation_1 = layers.Activation('relu')
        self._maxpooling2d = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')

        self._conv_block_1 = ConvBlock(3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        self._identity_block_1 = IdentityBlock(3, [64, 64, 256], stage=2, block='b')
        self._identity_block_2 = IdentityBlock(3, [64, 64, 256], stage=2, block='c')

        self._conv_block_2 = ConvBlock(3, [128, 128, 512], stage=3, block='a')
        self._identity_block_3 = IdentityBlock(3, [128, 128, 512], stage=3, block='b')
        self._identity_block_4 = IdentityBlock(3, [128, 128, 512], stage=3, block='c')
        self._identity_block_5 = IdentityBlock(3, [128, 128, 512], stage=3, block='d')

        self._conv_block_3 = ConvBlock(3, [256, 256, 1024], stage=4, block='a')
        self._identity_block_6 = IdentityBlock(3, [256, 256, 1024], stage=4, block='b')
        self._identity_block_7 = IdentityBlock(3, [256, 256, 1024], stage=4, block='c')
        self._identity_block_8 = IdentityBlock(3, [256, 256, 1024], stage=4, block='d')
        self._identity_block_9 = IdentityBlock(3, [256, 256, 1024], stage=4, block='e')
        self._identity_block_10 = IdentityBlock(3, [256, 256, 1024], stage=4, block='f')

        self._conv_block_4 = ConvBlock(3, [512, 512, 2048], stage=5, block='a')
        self._identity_block_11 = IdentityBlock(3, [512, 512, 2048], stage=5, block='b')
        self._identity_block_12 = IdentityBlock(3, [512, 512, 2048], stage=5, block='c')

        rm_axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
        self._lamba_2 = layers.Lambda(lambda x: backend.mean(x, rm_axes), name='reduce_mean')
        self._dense = layers.Dense(
            num_classes,
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            name='fc1000')
        self._activation_2 = layers.Activation('softmax')


    def call(self, inputs, training=False):
        if backend.image_data_format() == 'channels_first':
            x = self._lambda(inputs)
        else:
            x = inputs
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
        x = backend.cast(x, 'float32')
        return self._activation_2(x)


class IdentityBlock(tf.keras.Model):

    def __init__(self, kernel_size, filters, stage, block):
        super(IdentityBlock, self).__init__(name="identity_block")
        filters1, filters2, filters3 = filters
        if backend.image_data_format() == 'channels_last':
          bn_axis = 3
        else:
          bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self._conv2d_1 = layers.Conv2D(filters1, (1, 1), use_bias=False,
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                        name=conv_name_base + '2a')
        self._bn_1 = layers.BatchNormalization(axis=bn_axis,
                                    momentum=BATCH_NORM_DECAY,
                                    epsilon=BATCH_NORM_EPSILON,
                                    name=bn_name_base + '2a')
        self.activation_1 = layers.Activation('relu')

        self._conv2d_2 = layers.Conv2D(filters2, kernel_size,
                        padding='same', use_bias=False,
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                        name=conv_name_base + '2b')
        self._bn_2 = layers.BatchNormalization(axis=bn_axis,
                                    momentum=BATCH_NORM_DECAY,
                                    epsilon=BATCH_NORM_EPSILON,
                                    name=bn_name_base + '2b')
        self._activation_2 = layers.Activation('relu')

        self._conv2d_3 = layers.Conv2D(filters3, (1, 1), use_bias=False,
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                        name=conv_name_base + '2c')
        self._bn_3 = layers.BatchNormalization(axis=bn_axis,
                                    momentum=BATCH_NORM_DECAY,
                                    epsilon=BATCH_NORM_EPSILON,
                                    name=bn_name_base + '2c')

        self._activation_e3 = layers.Activation('relu')

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
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self._con2d_1 = layers.Conv2D(filters1, (1, 1), use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                          name=conv_name_base + '2a')
        self._bn_1 = layers.BatchNormalization(axis=bn_axis,
                                      momentum=BATCH_NORM_DECAY,
                                      epsilon=BATCH_NORM_EPSILON,
                                      name=bn_name_base + '2a')
        self._activation_1 = x = layers.Activation('relu')

        self._conv2d_2 = layers.Conv2D(filters2, kernel_size, strides=strides, padding='same',
                          use_bias=False, kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                          name=conv_name_base + '2b')
        self._bn_2 = layers.BatchNormalization(axis=bn_axis,
                                      momentum=BATCH_NORM_DECAY,
                                      epsilon=BATCH_NORM_EPSILON,
                                      name=bn_name_base + '2b')
        self._activation_2 = layers.Activation('relu')

        self._conv2d_3 = layers.Conv2D(filters3, (1, 1), use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                          name=conv_name_base + '2c')
        self._bn_3 = layers.BatchNormalization(axis=bn_axis,
                                      momentum=BATCH_NORM_DECAY,
                                      epsilon=BATCH_NORM_EPSILON,
                                      name=bn_name_base + '2c')

        self._shortcut = layers.Conv2D(filters3, (1, 1), strides=strides, use_bias=False,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                                 name=conv_name_base + '1')
        self._bn_4 = layers.BatchNormalization(axis=bn_axis,
                                             momentum=BATCH_NORM_DECAY,
                                             epsilon=BATCH_NORM_EPSILON,
                                             name=bn_name_base + '1')

        self._activation_4 = layers.Activation('relu')

    def call(self, inputs, training=False):
        x = self._con2d_1(inputs)
        x = self._bn_1(x)
        x = self._activation_1(x)
        x = self._conv2d_2(x)
        x = self._bn_2(x)
        x = self._activation_2(x)
        x = self._conv2d_3(x)
        x = self._conv2d_3(x)
        x = self._bn_3(x)
        shortcut = self._shortcut(inputs)
        shortcut = self._bn_4(shortcut)
        x = layers.add([x, shortcut])
        return self._activation_4(x) 
