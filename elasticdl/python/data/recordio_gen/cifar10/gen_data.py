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

    # Initilize codec
    codec_module = load_module(args.codec_file)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    train_example_list = numpy_to_examples(x_train, y_train)
    convert_examples_to_recordio(
        train_example_list,
        args.dir + "/cifar10/train",
        records_per_file=records_per_file,
        codec=codec_module.codec,
    )

    # Work around a bug in cifar10.load_data() where y_test is not converted
    # to uint8
    y_test = y_test.astype("uint8")
    test_example_list = numpy_to_examples(x_test, y_test)
    convert_examples_to_recordio(
        test_example_list,
        args.dir + "/cifar10/test",
        records_per_file=records_per_file,
        codec=codec_module.codec,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
