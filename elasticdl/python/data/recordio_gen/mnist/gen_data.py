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
        default=1024,
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
        "--fashion_mnist",
        action='store_true',
        help="If convert the Fashion MNIST dataset",
    )
    args = parser.parse_args(argv)

    records_per_file = args.num_record_per_chunk * args.num_chunk

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    feature_columns = [
        tf.feature_column.numeric_column(
            key="image", dtype=tf.float32, shape=[28, 28]
        ),
        tf.feature_column.numeric_column(
            key="label", dtype=tf.int64, shape=[1]
        ),
    ]

    # Initilize codec
    codec_module = load_module(args.codec_file)
    codec_module.codec.init(feature_columns)

    convert_numpy_to_recordio(
        args.dir + "/mnist/train",
        x_train,
        y_train,
        feature_columns,
        records_per_file=records_per_file,
        codec=codec_module.codec,
    )

    convert_numpy_to_recordio(
        args.dir + "/mnist/test",
        x_test,
        y_test,
        feature_columns,
        records_per_file=records_per_file,
        codec=codec_module.codec,
    )

    if args.fashion_mnist:
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        convert_numpy_to_recordio(
            args.dir + "/fashion/train",
            x_train,
            y_train,
            feature_columns,
            records_per_file=records_per_file,
            codec=codec_module.codec,
        )
        convert_numpy_to_recordio(
            args.dir + "/fashion/test",
            x_test,
            y_test,
            feature_columns,
            records_per_file=records_per_file,
            codec=codec_module.codec,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
