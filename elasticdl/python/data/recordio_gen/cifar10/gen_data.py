#!/usr/bin/env python

"""
Download and transform CIFAR10 data to RecordIO format.
"""

import argparse
import sys
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras.datasets import cifar10
from elasticdl.python.data.recordio_gen.convert_numpy_to_recordio import \
    convert_numpy_to_recordio


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate CIFAR10 datasets in RecordIO format."
    )
    parser.add_argument(
        "dir",
        help="Output directory",
    )
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
        "--codec_type",
        default="bytes",
        choices=["tf_example", "bytes"],
        help="Type of codec(tf_example or bytes)",
    )
    args = parser.parse_args(argv)

    record_per_file = args.num_record_per_chunk * args.num_chunk
    backend.set_image_data_format("channels_last")

    feature_columns = [
        tf.feature_column.numeric_column(
            key="image", dtype=tf.float32, shape=[32, 32, 3]
        ),
        tf.feature_column.numeric_column(
            key="label", dtype=tf.int64, shape=[1]
        ),
    ]

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    convert_numpy_to_recordio(
        args.dir + "/cifar10/train",
        x_train,
        y_train,
        feature_columns,
        record_per_file=record_per_file,
        codec_type=args.codec_type,
    )

    # Work around a bug in cifar10.load_data() where y_test is not converted
    # to uint8
    y_test = y_test.astype("uint8")
    convert_numpy_to_recordio(
        args.dir + "/cifar10/test",
        x_test,
        y_test,
        feature_columns,
        record_per_file=record_per_file,
        codec_type=args.codec_type,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
