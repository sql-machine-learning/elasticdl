#!/usr/bin/env python

"""
Download and transform CIFAR10 data to RecordIO format.
"""

import argparse
import sys
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras.datasets import cifar10
from elasticdl.python.data.recordio_gen.convert_numpy_to_recordio import (
    convert_numpy_to_recordio,
)
from elasticdl.python.elasticdl.common.model_helper import load_module


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate CIFAR10 datasets in RecordIO format."
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
    args = parser.parse_args(argv)

    records_per_file = args.num_record_per_chunk * args.num_chunk
    backend.set_image_data_format("channels_first")

    feature_columns = [
        tf.feature_column.numeric_column(
            key="image", dtype=tf.float32, shape=[32, 32, 3]
        ),
        tf.feature_column.numeric_column(
            key="label", dtype=tf.int64, shape=[1]
        ),
    ]

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Initilize codec
    codec_module = load_module(args.codec_file)
    codec_module.codec.init(feature_columns)

    convert_numpy_to_recordio(
        args.dir + "/cifar10/train",
        x_train,
        y_train,
        feature_columns,
        records_per_file=records_per_file,
        codec=codec_module.codec,
    )

    # Work around a bug in cifar10.load_data() where y_test is not converted
    # to uint8
    y_test = y_test.astype("uint8")
    convert_numpy_to_recordio(
        args.dir + "/cifar10/test",
        x_test,
        y_test,
        feature_columns,
        records_per_file=records_per_file,
        codec=codec_module.codec,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
