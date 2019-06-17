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
        "--codec_type",
        default="bytes",
        choices=["tf_example", "bytes"],
        help="Type of codec(tf_example or bytes)",
    )
    args = parser.parse_args(argv)

    records_per_file = args.num_record_per_chunk * args.num_chunk

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
    }

    feature_columns = [
        tf.feature_column.numeric_column(
            key="image", dtype=tf.float32, shape=[28, 28]
        ),
        tf.feature_column.numeric_column(
            key="label", dtype=tf.int64, shape=[1]
        ),
    ]


    # feature_columns = [
    #     tf.feature_column.numeric_column(
    #         key="image", dtype=tf.float32, shape=[28, 28]
    #     ),
    #     tf.feature_column.numeric_column(
    #         key="label", dtype=tf.int64, shape=[1]
    #     ),
    # ]
    convert_numpy_to_recordio(
        args.dir + "/mnist/train",
        x_train,
        y_train,
        feature_columns,
        records_per_file=records_per_file,
        codec_type=args.codec_type,
    )

    convert_numpy_to_recordio(
        args.dir + "/mnist/test",
        x_test,
        y_test,
        feature_columns,
        records_per_file=records_per_file,
        codec_type=args.codec_type,
    )

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    convert_numpy_to_recordio(
        args.dir + "/fashion/train",
        x_train,
        y_train,
        feature_columns,
        records_per_file=records_per_file,
        codec_type=args.codec_type,
    )
    convert_numpy_to_recordio(
        args.dir + "/fashion/test",
        x_test,
        y_test,
        feature_columns,
        records_per_file=records_per_file,
        codec_type=args.codec_type,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
