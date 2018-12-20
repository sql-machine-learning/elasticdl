#!/usr/bin/env python

"""
Download and transform MNIST and Fashion-MNIST data to RecordIO format.
"""

import itertools
import argparse
import os
import sys
from recordio import File
from tensorflow.python.keras.datasets import mnist, fashion_mnist
import record


def gen(file_dir, data, label, *, chunk_size, record_per_file):
    assert len(data) == len(label) and len(data) > 0
    os.makedirs(file_dir)
    it = zip(data, label)
    try:
        for i in itertools.count():
            file_name = file_dir + "/data-%04d" % i
            print("writing:", file_name)
            with File(file_name, "w", max_chunk_size=chunk_size) as f:
                for _ in range(record_per_file):
                    row = next(it)
                    f.write(record.encode(row[0], row[1]))
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
    )

    gen(
        args.dir + "/mnist/test",
        x_test,
        y_test,
        chunk_size=chunk_size,
        record_per_file=record_per_file,
    )

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    gen(
        args.dir + "/fashion/train",
        x_train,
        y_train,
        chunk_size=chunk_size,
        record_per_file=record_per_file,
    )
    gen(
        args.dir + "/fashion/test",
        x_test,
        y_test,
        chunk_size=chunk_size,
        record_per_file=record_per_file,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
