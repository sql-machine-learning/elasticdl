#!/usr/bin/env python

import argparse
import tensorflow as tf
import os
import recordio
import sys


def main(args):
    if args.dataset == "mnist":
        from tf.python.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif args.dataset == "fashion_mnist":
        from tf.python.keras.datasets import fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif args.dataset == "cifar10":
        from tf.python.keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    else:
        sys.exit("Unknown dataset {}".format(args.dataset))

    row = 0
    shard = 0
    w = None
    while row < x_train.shape[0] * args.fraction:
        if row % args.records_per_shard == 0:
            if w:
                w.close()
            fn = os.path.join(args.dir, "%s-%05d" % (args.dataset, shard))
            print("Writing {} ...".format(fn))
            w = recordio.Writer(fn)
            shard = shard + 1

        w.write(tf.train.Example(features=tf.train.Features(feature={
            "image": tf.train.Feature(float_list=tf.train.FloatList(
                value=x_train[row].flatten())),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(
                value=y_train[row].flatten()))})).SerializeToString())
        row = row + 1
    w.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(
        "Generate MNIST and Fashion-MNIST datasets in RecordIO format."))
    parser.add_argument("dir", help="Output directory")
    parser.add_argument("--records_per_shard", default=16 * 1024, type=int,
                        help="Maximum number of records per shard file.")
    parser.add_argument("--dataset", default="mnist",
                        help="Dataset name: mnist or fashion_mnist or cifar10")
    parser.add_argument("--fraction", default=1.0, type=float,
                        help="The fraction of the dataset to be converted")
    args = parser.parse_args(sys.argv[1:])
    main(args)
