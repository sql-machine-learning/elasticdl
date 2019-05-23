from recordio import File
from data.codec import TFExampleCodec
from data.codec import BytesCodec
import sys
import argparse
import tensorflow as tf
tf.enable_eager_execution()

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
        default="bytes",
        choices=["tf_example", "bytes"],
        help="Type of codec(tf_example or bytes)",
    )
    args = parser.parse_args(argv)

    feature_columns = [tf.feature_column.numeric_column(key="image",
        dtype=tf.float32, shape=[3, 32, 32]),
        tf.feature_column.numeric_column(key="label",
        dtype=tf.int64, shape=[1, 1])]
    if args.codec_type == "tf_example":
        decode_fn = TFExampleCodec(feature_columns).decode
    elif args.codec_type == "bytes":
        decode_fn = BytesCodec(feature_columns).decode
    else:
        raise ValueError("invalid codec_type: " + codec_type)
    with File(args.file, "r", decoder=decode_fn) as f:
        for i in range(
            args.start, args.start + (args.n * args.step), args.step
        ):
            print("-" * 10)
            print("record:", i)
            if args.codec_type == "tf_example":
                print(f.get(i)['image'].numpy())
                print(f.get(i)['label'].numpy())
            elif args.codec_type == "bytes":
                print(f.get(i))


if __name__ == "__main__":
    main(sys.argv[1:])
