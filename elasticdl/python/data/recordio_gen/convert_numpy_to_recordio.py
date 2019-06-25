#!/usr/bin/env python

"""
Convert data in numpy format to RecordIO format.
"""
import itertools
import os
from contextlib import closing
import recordio


def convert_numpy_to_recordio(
    file_dir,
    data,
    label,
    feature_columns,
    records_per_file,
    codec,
    partition="",
):
    """
    Convert data in numpy format to RecordIO format
    """
    assert len(data) > 0
    assert len(data) == len(label)

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    feature_name_to_type = {
        f_col.key: f_col.dtype for f_col in feature_columns
    }

    it = zip(data, label)
    try:
        for i in itertools.count():
            if partition == "":
                file_name = file_dir + "/data-%04d" % i
            else:
                file_name = file_dir + "/data-%s-%04d" % (partition, i)
            print("writing:", file_name)

            encode_fn = codec.encode
            with closing(recordio.Writer(file_name)) as f:
                for _ in range(records_per_file):
                    row = next(it)
                    example = {
                        f_col.key: row[i]
                        .astype(f_col.dtype.as_numpy_dtype)
                        .reshape(f_col.shape)
                        for i, f_col in enumerate(feature_columns)
                    }
                    rec = encode_fn(example, feature_name_to_type)
                    f.write(rec)
    except StopIteration:
        pass
