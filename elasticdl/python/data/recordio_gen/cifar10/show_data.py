import sys
import argparse
import recordio
import tensorflow as tf

from contextlib import closing
from elasticdl.python.data.codec import TFExampleCodec
from elasticdl.python.data.codec import BytesCodec


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
        "--codec_type",
        default="bytes",
        choices=["tf_example", "bytes"],
        help="Type of codec(tf_example or bytes)",
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
    if args.codec_type == "tf_example":
        decode_fn = TFExampleCodec(feature_columns).decode
    elif args.codec_type == "bytes":
        decode_fn = BytesCodec(feature_columns).decode
    else:
        raise ValueError("invalid codec_type: " + args.codec_type)

    with closing(recordio.Scanner(args.file, args.start, args.n)) as f:
        for i in range(args.start, args.start + args.n):
            rec = f.record()
            if rec is None:
                break
            rec = decode_fn(rec)

            print("-" * 10)
            print("record:", i)
            if args.codec_type == "tf_example":
                print(rec["image"].numpy())
                print(rec["label"].numpy())
            elif args.codec_type == "bytes":
                print(rec)


if __name__ == "__main__":
    main(sys.argv[1:])
