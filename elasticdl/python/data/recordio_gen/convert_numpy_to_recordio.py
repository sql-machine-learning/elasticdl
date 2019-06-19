#!/usr/bin/env python

"""
Convert data in numpy format to RecordIO format.
"""
import itertools
import os
from contextlib import closing
import recordio


def convert_examples_to_recordio(
    file_dir,
    examples,
    # data,
    # label,
    # feature_columns,
    records_per_file,
    encode_fn,
    partition="",
):
    """
    Convert data in numpy format to RecordIO format
    """
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    try:
        for i in itertools.count():
            if partition == "":
                file_name = file_dir + "/data-%04d" % i
            else:
                file_name = file_dir + "/data-%s-%04d" % (partition, i)
            print("writing:", file_name)

            with closing(recordio.Writer(file_name)) as f:
                for _ in range(records_per_file):
                    for example in examples:
                        f.write(encode_fn(example))
    except StopIteration:
        pass
