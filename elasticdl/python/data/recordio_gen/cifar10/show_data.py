import sys
import argparse
import recordio
import tensorflow as tf

from contextlib import closing
from elasticdl.python.elasticdl.common.model_helper import load_module


# TODO: share code with MNIST dataset.
def main(argv):
    parser = argparse.ArgumentParser(
        description="Show some data from CIFAR10 recordio"
    )
    parser.add_argument("file", help="RecordIo file to read")
    parser.add_argument(
        "--start", default=0, type=int, help="Start record number"
    )
    parser.add_argument(
        "--n", default=20, type=int, help="How many record to show"
    )
    parser.add_argument(
        "--codec_file",
        default="elasticdl/python/data/codec/tf_example_codec.py",
        help="Codec file name",
    )
    args = parser.parse_args(argv)

    feature_columns = [
        tf.feature_column.numeric_column(
            key="image", dtype=tf.float32, shape=[32, 32, 3]
        ),
        tf.feature_column.numeric_column(
            key="label", dtype=tf.int64, shape=[1]
        ),
    ]
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    # Initilize codec
    codec_module = load_module(args.codec_file)
    decode_fn = codec_module.codec.decode

    with closing(recordio.Scanner(args.file, args.start, args.n)) as f:
        for i in range(args.start, args.start + args.n):
            rec = f.record()
            if rec is None:
                break
            rec = decode_fn(rec, feature_spec)

            print("-" * 10)
            print("record:", i)
            print(rec["image"].numpy())
            print(rec["label"].numpy())


if __name__ == "__main__":
    main(sys.argv[1:])
