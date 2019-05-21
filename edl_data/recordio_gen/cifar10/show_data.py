from recordio import File
from edl_data.codec import TFExampleCodec
from edl_data.codec import BytesCodec
import sys
import argparse
import tensorflow as tf

# TODO: share code with MNIST dataset.
def main(argv):
    print(argv)
    parser = argparse.ArgumentParser(
        description="Show same data from CIFAR10 recordio"
    )
    parser.add_argument("file", help="RecordIo file to read")
    parser.add_argument(
        "--start", default=0, type=int, help="Start record number"
    )
    parser.add_argument("--step", default=1, type=int, help="Step")
    parser.add_argument(
        "--n", default=20, type=int, help="How many record to show"
    )
    parser.add_argument(
        "--codec_type",
        default=None,
        choices=["tf_example"],
        help="Type of codec(tf_example or None)",
    )
    args = parser.parse_args(argv)

    feature_columns = [tf.feature_column.numeric_column(key="image",
        dtype=tf.float32, shape=[1, 32, 32]),
        tf.feature_column.numeric_column(key="label",
        dtype=tf.int64, shape=[1])]
    if args.codec_type == "tf_example":
        decode_fn = TFExampleCodec(feature_columns).decode
    else:
        decode_fn = BytesCodec(feature_columns).decode
    with File(args.file, "r", decoder=decode_fn) as f:
        for i in range(
            args.start, args.start + (args.n * args.step), args.step
        ):
            print("-" * 10)
            print("record:", i)
            print(f.get(i))


if __name__ == "__main__":
    main(sys.argv[1:])
