#!/usr/bin/env python

import argparse
import logging
import os
import sys

import recordio
import tensorflow as tf
import numpy as np
import LoadData

urls = {
    "train": "https://raw.githubusercontent.com/hexiangnan/neural_factorization_machine/master/data/frappe/frappe.train.libfm",
    "validation": "https://raw.githubusercontent.com/hexiangnan/neural_factorization_machine/master/data/frappe/frappe.validation.libfm",
    "test": "https://raw.githubusercontent.com/hexiangnan/neural_factorization_machine/master/data/frappe/frappe.test.libfm"
}


def convert(x, y, args, subdir):
    """Convert pairs of feature and label in NumPy arrays into a set of
    RecordIO files.
    """
    row = 0
    shard = 0
    w = None
    while row < x.shape[0] * args.fraction:
        if row % args.records_per_shard == 0:
            if w:
                w.close()
            dn = os.path.join(args.output_dir, subdir)
            fn = os.path.join(dn, "data-%05d" % (shard))
            if not os.path.exists(dn):
                os.makedirs(os.path.dirname(fn))
            logging.info("Writing {} ...".format(fn))
            w = recordio.Writer(fn)
            shard = shard + 1

        w.write(
            tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "feature": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=x[row].flatten()
                            )
                        ),
                        "label": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=y[row].flatten()
                            )
                        ),
                    }
                )
            ).SerializeToString()
        )
        row = row + 1
    w.close()
    print('done')
    logging.info(
        "Wrote {} of total {} records into {} files".format(
            row, x.shape[0], shard
        )
    )

def load_raw_data(args):
    for phase in ["train", "validation", "test"]:
        filepath = os.path.join(args.data, phase + ".libfm")
        if not os.path.exists(filepath):
            download(urls[phase], filepath)
    data = LoadData.LoadData(args.data)
    return (
        data.train[0], 
        data.train[1], 
        data.validation[0],
        data.validation[1],
        data.test[0],
        data.test[1]
    )
        
def main(args):
    x_train, y_train, x_val, y_val, x_test, y_test = load_raw_data(args)

    convert(x_train, y_train, args, "train")
    convert(x_val, y_val, args, "val")
    convert(x_test, y_test, args, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Convert TensorFlow feature datasets into RecordIO format.")
    )
    parser.add_argument("--data", help="Data file path")
    parser.add_argument(
        "--records_per_shard",
        default=16 * 1024,
        type=int,
        help="Maximum number of records per shard file.",
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory",
    )
    parser.add_argument(
        "--fraction",
        default=1.0,
        type=float,
        help="The fraction of the dataset to be converted",
    )
    args = parser.parse_args(sys.argv[1:])
    main(args)
