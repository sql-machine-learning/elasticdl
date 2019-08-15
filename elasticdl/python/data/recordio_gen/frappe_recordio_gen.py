#!/usr/bin/env python

import argparse
import logging
import os
import sys

import numpy as np
import recordio
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

urls_template = (
    "https://raw.githubusercontent.com/hexiangnan/neural_factorization"
    "_machine/master/data/frappe/frappe.%s.libfm"
)

urls = {
    "train": urls_template % "train",
    "validation": urls_template % "validation",
    "test": urls_template % "test",
}


class LoadFrappe(object):
    def __init__(self, path):
        self.trainfile = os.path.join(path, "train.libfm")
        self.testfile = os.path.join(path, "test.libfm")
        self.validationfile = os.path.join(path, "validation.libfm")
        self.feature_num = self.gen_feature_map()
        print("feature_num:%d" % self.feature_num)

        self.train = self.read_data(self.trainfile)
        maxlen_train = max([len(i) for i in self.train[0]])

        self.validation = self.read_data(self.validationfile)
        maxlen_val = max([len(i) for i in self.validation[0]])

        self.test = self.read_data(self.testfile)
        maxlen_test = max([len(i) for i in self.test[0]])

        self.maxlen = max(maxlen_train, maxlen_val, maxlen_test)
        print("maxlen:%d" % self.maxlen)

        self.train = self.to_numpy(self.train, self.maxlen)
        self.validation = self.to_numpy(self.validation, self.maxlen)
        self.test = self.to_numpy(self.test, self.maxlen)

    def gen_feature_map(self):
        self.features = {}
        self._read_features(self.trainfile)
        self._read_features(self.testfile)
        self._read_features(self.validationfile)
        return len(self.features) + 1

    def _read_features(self, filepath):
        with open(filepath, "r") as fp:
            for line in fp:
                for item in line.strip().split(" ")[1:]:
                    # 0 for pad_sequences
                    self.features.setdefault(item, len(self.features) + 1)

    def read_data(self, datafile):
        x, y = [], []
        with open(datafile, "r") as fp:
            for line in fp:
                arr = line.strip().split(" ")
                if float(arr[0]) > 0:
                    y.append(1)
                else:
                    y.append(0)
                x.append([self.features[item] for item in arr[1:]])
        return x, y

    def to_numpy(self, data, maxlen):
        x, y = data
        maxlen = max([len(i) for i in x])
        x = pad_sequences(x, maxlen=maxlen)
        return (np.array(x, dtype=np.int64), np.array(y, dtype=np.int64))


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
    print("done")
    logging.info(
        "Wrote {} of total {} records into {} files".format(
            row, x.shape[0], shard
        )
    )


def download(url, dest_path):

    req = requests.get(url, stream=True)
    req.raise_for_status()

    with open(dest_path, "wb") as fd:
        for chunk in req.iter_content(chunk_size=2 ** 20):
            fd.write(chunk)


def load_raw_data(args):
    for phase in ["train", "validation", "test"]:
        filepath = os.path.join(args.data, phase + ".libfm")
        if not os.path.exists(filepath):
            download(urls[phase], filepath)
    data = LoadFrappe(args.data)
    return (
        data.train[0],
        data.train[1],
        data.validation[0],
        data.validation[1],
        data.test[0],
        data.test[1],
    )


def main(args):
    x_train, y_train, x_val, y_val, x_test, y_test = load_raw_data(args)

    convert(x_train, y_train, args, "train")
    convert(x_val, y_val, args, "val")
    convert(x_test, y_test, args, "test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert TensorFlow feature datasets into RecordIO format."
        )
    )
    parser.add_argument("--data", help="Data file path")
    parser.add_argument(
        "--records_per_shard",
        default=16 * 1024,
        type=int,
        help="Maximum number of records per shard file.",
    )
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument(
        "--fraction",
        default=1.0,
        type=float,
        help="The fraction of the dataset to be converted",
    )
    args = parser.parse_args(sys.argv[1:])
    main(args)
