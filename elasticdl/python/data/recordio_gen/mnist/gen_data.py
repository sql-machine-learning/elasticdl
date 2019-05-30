#!/usr/bin/env python

"""
Download and transform MNIST and Fashion-MNIST data to RecordIO format.
"""

import itertools
import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import recordio

from contextlib import closing
from elasticdl.python.data.codec import TFExampleCodec
from elasticdl.python.data.codec import BytesCodec
from tensorflow.python.keras.datasets import mnist, fashion_mnist


def gen(file_dir, data, label, *, chunk_size, record_per_file, codec_type):
    assert len(data) == len(label) and len(data) > 0
    os.makedirs(file_dir)
    it = zip(data, label)
    try:
        for i in itertools.count():
            file_name = file_dir + "/data-%04d" % i
            print("writing:", file_name)
            feature_columns = [
                tf.feature_column.numeric_column(
                    key="image", dtype=tf.float32, shape=[28, 28]
                ),
                tf.feature_column.numeric_column(
                    key="label", dtype=tf.int64, shape=[1]
                ),
            ]
            if codec_type == "tf_example":
                encode_fn = TFExampleCodec(feature_columns).encode
            elif codec_type == "bytes":
                encode_fn = BytesCodec(feature_columns).encode
            else:
                raise ValueError("invalid codec_type: " + codec_type)
            with closing(recordio.Writer(file_name)) as f:
                for _ in range(record_per_file):
                    row = next(it)
                    rec = encode_fn(
                        {
                            f_col.key: row[i]
                            .astype(f_col.dtype.as_numpy_dtype)
                            .reshape(f_col.shape)
                            for i, f_col in enumerate(feature_columns)
                        }
                    )
                    f.write(rec)
    except StopIteration:
        pass


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate MNIST and Fashion-MNIST datasets in RecordIO format."
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
    # one uncompressed record has size 28 * 28 + 1 bytes.
    # Also add some slack for safety.
    chunk_size = args.num_record_per_chunk * (28 * 28 + 1) + 100
    record_per_file = args.num_record_per_chunk * args.num_chunk

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    gen(
        args.dir + "/mnist/train",
        x_train,
        y_train,
        chunk_size=chunk_size,
        record_per_file=record_per_file,
        codec_type=args.codec_type,
    )

    gen(
        args.dir + "/mnist/test",
        x_test,
        y_test,
        chunk_size=chunk_size,
        record_per_file=record_per_file,
        codec_type=args.codec_type,
    )

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    gen(
        args.dir + "/fashion/train",
        x_train,
        y_train,
        chunk_size=chunk_size,
        record_per_file=record_per_file,
        codec_type=args.codec_type,
    )
    gen(
        args.dir + "/fashion/test",
        x_test,
        y_test,
        chunk_size=chunk_size,
        record_per_file=record_per_file,
        codec_type=args.codec_type,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
