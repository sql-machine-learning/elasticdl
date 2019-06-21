#!/usr/bin/env python

"""
Download and transform MNIST and Fashion-MNIST data to RecordIO format.
"""

import argparse
import sys
import tensorflow as tf
from tensorflow.python.keras.datasets import mnist, fashion_mnist

from elasticdl.python.data.recordio_gen.convert_numpy_to_recordio import (
    convert_examples_to_recordio,
)
from elasticdl.python.elasticdl.common.model_helper import load_module


def numpy_to_examples(x, y):
    train_example_list = []
    assert len(x) == len(y)
    for i in range(len(x)):
        numpy_image = x[i]
        label = y[i]
        feature_name_to_feature = {}
        feature_name_to_feature['image'] = tf.train.Feature(
            float_list=tf.train.FloatList(
                value=numpy_image.astype(tf.float32.as_numpy_dtype).flatten(),
            ),
        )
        feature_name_to_feature['label'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[label]),
        )
        example = tf.train.Example(
            features=tf.train.Features(feature=feature_name_to_feature),
        )
        train_example_list.append(example)
    return train_example_list


def main(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Generate MNIST and Fashion-MNIST datasets in RecordIO format."
        )
    )
    parser.add_argument("dir", help="Output directory")
    parser.add_argument(
        "--num_record_per_chunk",
        default=256,
        type=int,
        help="Approximate number of records in a chunk.",
    )
    parser.add_argument(
        "--num_chunk",
        default=16,
        type=int,
        help="Number of chunks in a RecordIO file",
    )
    parser.add_argument(
        "--codec_file",
        default="elasticdl/python/data/codec/tf_example_codec.py",
        help="Codec file name",
    )
    parser.add_argument(
        "--mnist_fraction",
        default=1.0,            # 100%
        type=float,
        help="The fraction of the MNIST dataset to be converted",
    )
    parser.add_argument(
        "--fashion_mnist_fraction",
        default=0.0,            # 0%
        type=float,
        help="The fraction of the Fashion MNIST dataset to be converted",
    )
    args = parser.parse_args(argv)

    records_per_file = args.num_record_per_chunk * args.num_chunk

    # Initilize codec
    codec_module = load_module(args.codec_file)

    n = max(0.0, min(1.0, args.mnist_fraction))
    if n > 0:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        n = round(x_train.shape[0] * n)
        train_example_list = numpy_to_examples(x_train[:n], y_train[:n])
        convert_examples_to_recordio(
            args.dir + "/mnist/train",
            train_example_list,
            records_per_file=records_per_file,
            encode_fn=codec_module.codec.encode,
        )

        n = round(x_train.shape[0] * n)
        test_example_list = numpy_to_examples(x_test[:n], y_test[:n])
        convert_examples_to_recordio(
            args.dir + "/mnist/test",
            test_example_list,
            records_per_file=records_per_file,
            encode_fn=codec_module.codec.encode,
        )

    n = max(0.0, min(1.0, args.fashion_mnist_fraction))
    if n > 0:
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        n = round(x_train.shape[0] * n)
        train_example_list = numpy_to_examples(x_train[:n], y_train[:n])
        convert_examples_to_recordio(
            args.dir + "/fashion/train",
            train_example_list,
            records_per_file=records_per_file,
            encode_fn=codec_module.codec.encode,
        )

        n = round(x_train.shape[0] * n)
        test_example_list = numpy_to_examples(x_test[:n], y_test[:n])
        convert_examples_to_recordio(
            args.dir + "/fashion/test",
            test_example_list,
            records_per_file=records_per_file,
            encode_fn=codec_module.codec.encode,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
