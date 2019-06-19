import recordio
import sys
import argparse

from contextlib import closing
from elasticdl.python.elasticdl.common.model_helper import load_module
import tensorflow as tf


def main(argv):
    parser = argparse.ArgumentParser(
        description="Show some data from mnist recordio"
    )
    parser.add_argument("file", help="RecordIo file to read")
    parser.add_argument(
        "--start", default=0, type=int, help="Start record number"
    )
    parser.add_argument(
        "--n", default=20, type=int, help="How many records to show"
    )
    parser.add_argument(
        "--codec_file",
        default="elasticdl/python/data/codec/tf_example_codec.py",
        help="Codec file name",
    )
    args = parser.parse_args(argv)

    feature_columns = [
        tf.feature_column.numeric_column(
            key="image", dtype=tf.float32, shape=[1, 28, 28]
        ),
        tf.feature_column.numeric_column(
            key="label", dtype=tf.int64, shape=[1]
        ),
    ]

    # Initilize codec
    codec_module = load_module(args.codec_file)
    codec_module.codec.init(feature_columns)

    decode_fn = codec_module.codec.decode

    with closing(recordio.Scanner(args.file, args.start, args.n)) as f:
        for i in range(args.start, args.start + args.n):
            rec = f.record()
            if rec is None:
                break
            rec = decode_fn(rec)

            print("-" * 10)
            print("record:", i)
            print(rec["image"].numpy())
            print(rec["label"].numpy())


if __name__ == "__main__":
    main(sys.argv[1:])
