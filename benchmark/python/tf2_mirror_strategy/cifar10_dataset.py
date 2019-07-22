import os
import pickle
from contextlib import closing

import numpy as np
import recordio
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


def process_dataset(
    dataset,
    batch_size,
    epoch,
    training=False,
    augmentation=False,
    shuffle=False,
):
    if training and augmentation:
        dataset = dataset.map(augmentation)
    dataset = dataset.map(
        lambda x, y: (
            {"image": tf.math.divide(tf.cast(x, tf.float32), 255.0)},
            y,
        )
    )
    dataset = dataset.batch(batch_size)
    if training and shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.repeat(epoch)
    dataset = dataset.prefetch(1)
    return dataset


def get_cifar10_train_dataset(batch_size, epoch, data_path=None):
    x_train, y_train = get_cifar10_train_data(data_path)
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    return process_dataset(dataset, batch_size, epoch, training=True)


def get_cifar10_test_dataset(batch_size, epoch, data_path=None):
    x_test, y_test = get_cifar10_test_data(data_path)
    dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return process_dataset(dataset, batch_size, epoch, training=False)


def recordio_dataset(recordio_shards):
    """
    recordio_shards is a list of (recordio_file_name, start, end) tuples
    """

    class _Generator:
        def __init__(self, recordio_shards):
            self._shards = recordio_shards

        def gen(self):
            for s in self._shards:
                with closing(recordio.Scanner(s[0], s[1], s[2])) as reader:
                    while True:
                        r = reader.record()
                        if r:
                            yield r
                        else:
                            break

    generator = _Generator(recordio_shards)
    ds = tf.data.Dataset.from_generator(
        generator.gen, (tf.string), (tf.TensorShape([]))
    )
    return ds


def recordio_dataset_from_dir(data_path):
    """
    data_path is the directory that includes all the recordio files
    """
    shards = []
    for f in os.listdir(data_path):
        p = os.path.join(data_path, f)
        with closing(recordio.Index(p)) as rio:
            shards.append((p, 0, rio.num_records()))
    return recordio_dataset(shards)


def parse_cifar10_data(record):
    feature_description = {
        "image": tf.io.FixedLenFeature([32, 32, 3], tf.float32),
        "label": tf.io.FixedLenFeature([1], tf.int64),
    }
    r = tf.io.parse_single_example(record, feature_description)
    label = r["label"]
    image = r["image"]
    return image, label


def get_cifar10_recordio_dataset(batch_size, epoch, recordio_shards, training):
    dataset = recordio_dataset(recordio_shards)
    dataset = dataset.map(parse_cifar10_data)
    return process_dataset(dataset, batch_size, epoch, training=training)


def get_cifar10_recordio_dataset_from_dir(
    batch_size, epoch, data_path, training
):
    dataset = recordio_dataset_from_dir(data_path)
    dataset = dataset.map(parse_cifar10_data)
    return process_dataset(dataset, batch_size, epoch, training=training)
