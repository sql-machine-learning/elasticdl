#!/usr/bin/env python

"""
Download and transform MNIST and Fashion-MNIST data to RecordIO format.
"""

import argparse
import sys
import tensorflow as tf
from tensorflow.python.keras.datasets import mnist, fashion_mnist

from elasticdl.python.data.recordio_gen.convert_numpy_to_recordio import (
    convert_numpy_to_recordio,
)
from elasticdl.python.elasticdl.common.model_helper import load_module


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

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    train_example_list = numpy_to_examples(x_train, y_train)
    convert_examples_to_recordio(
        args.dir + "/mnist/train",
        example,
        # x_train,
        # y_train,
        # feature_columns,
        records_per_file=records_per_file,
        codec=codec_module.codec,
    )

    convert_numpy_to_recordio(
        args.dir + "/mnist/test",
        example,
        # x_test,
        # y_test,
        # feature_columns,
        records_per_file=records_per_file,
        codec=codec_module.codec,
    )

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    convert_numpy_to_recordio(
        args.dir + "/fashion/train",
        example,
        # x_train,
        # y_train,
        # feature_columns,
        records_per_file=records_per_file,
        codec=codec_module.codec,
    )
    convert_numpy_to_recordio(
        args.dir + "/fashion/test",
        example,
        # x_test,
        # y_test,
        # feature_columns,
        records_per_file=records_per_file,
        codec=codec_module.codec,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
