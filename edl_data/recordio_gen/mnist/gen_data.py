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
from recordio import File
from edl_data.codec import TFExampleCodec
from edl_data.codec import BytesCodec
from tensorflow.python.keras.datasets import mnist, fashion_mnist


def gen(file_dir, data, label, *, chunk_size, record_per_file, codec_type):
    assert len(data) == len(label) and len(data) > 0
    os.makedirs(file_dir)
    it = zip(data, label)
    try:
        for i in itertools.count():
            file_name = file_dir + "/data-%04d" % i
            print("writing:", file_name)
            if codec_type == "tf_example":
                feature_columns = [tf.feature_column.numeric_column(key="image",
                    dtype=tf.float32, shape=[1, 28, 28]),
                    tf.feature_column.numeric_column(key="label",
                    dtype=tf.int64, shape=[1])]
                encode_fn = TFExampleCodec(feature_columns).encode
            elif codec_type == "bytes":
                encode_fn = BytesCodec(feature_columns).encode 
            else:
                raise ValueError("invalid codec_type: " + codec_type)
            with File(file_name, "w", max_chunk_size=chunk_size, encoder=encode_fn) as f:
                for _ in range(record_per_file):
                    row = next(it)
                    f.write([("image", row[0]), ("label", np.array([row[1]]))])
    except StopIteration:
        pass


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate MNIST and Fashion-MNIST datasets in RecordIO format."
    )
    parser.add_argument("dir", help="Output directory")
    parser.add_argument(
        "--num-record-per-chunk",
        default=1024,
        type=int,
        help="Approximate number of records in a chunk.",
    )
    parser.add_argument(
        "--num-chunk",
        default=16,
        type=int,
        help="Number of chunks in a RecordIO file",
    )
    parser.add_argument(
        "--codec_type",
        choices=["tf_example", "bytes"],
        help="Type of codec(tf_example or bytes)",
    )
    args = parser.parse_args(argv)
    # one uncompressed record has size 28 * 28 + 1 bytes.
    # Also add some slack for safety.
    chunk_size = args.num_record_per_chunk * (28 * 28  + 1) + 100
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
