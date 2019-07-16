import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.data_utils import get_file


def get_data(filename):
    """
    Return a tuple (data, labels)
    """
    with open(filename, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    data = dict[b"data"]
    labels = dict[b"labels"]
    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def get_path(path):
    if not path:
        dirname = "cifar-10-batches-py"
        origin = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        path = get_file(dirname, origin=origin, untar=True)
    return path


def get_cifar10_train_data(path=None):
    path = get_path(path)

    for i in range(1, 6):
        fpath = os.path.join(path, "data_batch_" + str(i))
        data, labels = get_data(fpath)
        if i == 1:
            x_train = data
            y_train = labels
        else:
            x_train = np.append(x_train, data)
            y_train = y_train + labels
    x_train = x_train.reshape(50000, 3, 32, 32)
    y_train = np.reshape(y_train, (len(y_train), 1))
    if K.image_data_format() == "channels_last":
        x_train = x_train.transpose(0, 2, 3, 1)

    return x_train, y_train


def get_cifar10_test_data(path=None):
    path = get_path(path)

    fpath = os.path.join(path, "test_batch")
    x_test, y_test = get_data(fpath)

    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == "channels_last":
        x_test = x_test.transpose(0, 2, 3, 1)

    return x_test, y_test


def augmentation(x, y):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x, y


def get_dataset(
    x, y, batch_size, epoch, training=False, augmentation=False, shuffle=False
):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if training and augmentation:
        dataset = dataset.map(augmentation)
    dataset = dataset.map(
        lambda x, y: (tf.math.divide(tf.cast(x, tf.float32), 255.0), y)
    )
    dataset = dataset.batch(batch_size)
    if training and shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.repeat(epoch)
    dataset = dataset.prefetch(1)
    return dataset


def get_cifar10_train_dataset(batch_size, epoch, data_path=None):
    x_train, y_train = get_cifar10_train_data(data_path)
    return get_dataset(x_train, y_train, batch_size, epoch, training=True)


def get_cifar10_test_dataset(batch_size, epoch, data_path=None):
    x_test, y_test = get_cifar10_test_data(data_path)
    return get_dataset(x_test, y_test, batch_size, epoch, training=False)
