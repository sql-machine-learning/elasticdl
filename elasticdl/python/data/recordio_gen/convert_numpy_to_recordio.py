#!/usr/bin/env python

"""
Convert data in numpy format to RecordIO format.
"""
import itertools
import os
from contextlib import closing
import recordio

from elasticdl.python.data.codec import TFExampleCodec
from elasticdl.python.data.codec import BytesCodec


def convert_numpy_to_recordio(
    file_dir,
    data,
    label,
    feature_columns,
    records_per_file,
    codec_type,
    partition="",
):
    """
    Convert data in numpy format to RecordIO format
    """
    assert len(data) > 0
    assert len(data) == len(label)

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    it = zip(data, label)
    try:
        for i in itertools.count():
            if partition == "":
                file_name = file_dir + "/data-%04d" % i
            else:
                file_name = file_dir + "/data-%s-%04d" % (partition, i)
            print("writing:", file_name)
            if codec_type == "tf_example":
                encode_fn = TFExampleCodec(feature_columns).encode
            elif codec_type == "bytes":
                encode_fn = BytesCodec(feature_columns).encode
            else:
                raise ValueError("invalid codec_type: " + codec_type)
            with closing(recordio.Writer(file_name)) as f:
                for _ in range(records_per_file):
                    row = next(it)
                    rec = encode_fn(
                        {
                            f_col.key: row[i]
                            .astype(f_col.dtype.as_numpy_dtype)
                            .reshape(f_col.shape)
                            for i, f_col in enumerate(feature_columns)
                        }
                    )
                    f.write(rec)
    except StopIteration:
        pass
