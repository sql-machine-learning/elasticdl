#!/usr/bin/env python

"""
Download and transform CIFAR10 data to RecordIO format.
"""

import itertools
import argparse
import os
import sys
from recordio import File
from tensorflow.python.keras import backend
from tensorflow.python.keras.datasets import cifar10
from edl_data.codec import TFExampleCodec
from edl_data.codec import BytesCodec
import tensorflow as tf
import numpy as np

# TODO: This function can be shared with MNIST dataset
def gen(file_dir, data, label, *, chunk_size, record_per_file, codec_type):
    assert len(data) == len(label) and len(data) > 0
    os.makedirs(file_dir)
    it = zip(data, label)
    try:
        for i in itertools.count():
            file_name = file_dir + "/data-%04d" % i
            print("writing:", file_name)
            feature_columns = [tf.feature_column.numeric_column(key="image",
                dtype=tf.float32, shape=[3, 32, 32]),
                tf.feature_column.numeric_column(key="label",
                dtype=tf.int64, shape=[1, 1])]
            if codec_type == "tf_example":
                encode_fn = TFExampleCodec(feature_columns).encode
            elif codec_type == "bytes":
                encode_fn = BytesCodec(feature_columns).encode
            else:
                raise ValueError("invalid codec_type: " + codec_type)
            with File(file_name, "w", max_chunk_size=chunk_size, encoder=encode_fn) as f:
                for _ in range(record_per_file):
                    row = next(it)
                    f.write([("image", row[0].astype(feature_columns[0].dtype.as_numpy_dtype)), 
                        ("label", np.array([row[1].astype(feature_columns[1].dtype.as_numpy_dtype)]))])
    except StopIteration:
        pass


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
        "--codec_type",
        choices=["tf_example", "bytes"],
        help="Type of codec(tf_example or bytes)",
    )
    args = parser.parse_args(argv)
    # one uncompressed record has size 3 * 32 * 32 + 1 bytes.
    # Also add some slack for safety.
    chunk_size = args.num_record_per_chunk * (3 * 32 * 32 + 1) + 100
    record_per_file = args.num_record_per_chunk * args.num_chunk
    backend.set_image_data_format("channels_first")

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    gen(
        args.dir + "/cifar10/train",
        x_train,
        y_train,
        chunk_size=chunk_size,
        record_per_file=record_per_file,
        codec_type=args.codec_type,
    )

    # Work around a bug in cifar10.load_data() where y_test is not converted
    # to uint8
    y_test = y_test.astype("uint8")
    gen(
        args.dir + "/cifar10/test",
        x_test,
        y_test,
        chunk_size=chunk_size,
        record_per_file=record_per_file,
        codec_type=args.codec_type,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
