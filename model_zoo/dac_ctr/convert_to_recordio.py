import argparse
import os
import pathlib
import sys

import recordio
import tensorflow as tf

DAC_COLUMNS = [
    "label",
    "I1",
    "I2",
    "I3",
    "I4",
    "I5",
    "I6",
    "I7",
    "I8",
    "I9",
    "I10",
    "I11",
    "I12",
    "I13",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
    "C22",
    "C23",
    "C24",
    "C25",
    "C26",
]

DAC_DTYPES = ["int64"] * 14 + ["str"] * 26


def convert_series_to_tf_feature(data_series, columns, dtype_series):
    """
    Convert pandas series to TensorFlow features.
    Args:
        data_series: Pandas series of data content.
        columns: Column name array.
        dtype_series: Pandas series of dtypes.
    Return:
        A dict of feature name -> tf.train.Feature
    """
    features = {}
    data_series = data_series.split("\t")
    for i, column_name in enumerate(columns):
        feature = None
        value = data_series[i]
        value = value.strip()
        dtype = dtype_series[i]

        if dtype == "int64":
            if value == "":
                value = 0
            else:
                value = int(value)
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[value])
            )
        elif dtype == "str":
            feature = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[value.encode("utf-8")])
            )
        else:
            assert False, "Unrecoginize dtype: {}".format(dtype)

        features[column_name] = feature

    return features


def convert_to_recordio_files(file_path, dir_name, records_per_shard):
    """
    Convert a pandas DataFrame to recordio files.
    Args:
        file_path: A path of the data file
        dir_name: A directory to put the generated recordio files.
        records_per_shard: The record number per shard.
    """
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)

    writer = None
    with open(file_path, "r") as f:
        for index, row in enumerate(f):
            if index % records_per_shard == 0:
                if writer:
                    writer.close()

                shard = index // records_per_shard
                file_path_name = os.path.join(dir_name, "data-%05d" % shard)
                writer = recordio.Writer(file_path_name)

            feature = convert_series_to_tf_feature(
                row, DAC_COLUMNS, DAC_DTYPES
            )
            result_string = tf.train.Example(
                features=tf.train.Features(feature=feature)
            ).SerializeToString()
            writer.write(result_string)

        if writer:
            writer.close()

        print("Finish data conversion in {}".format(dir_name))


def split_to_train_test(data_path):
    data_dir = os.path.dirname(data_path)

    train_file_name = os.path.join(data_dir, "edl_train.txt")
    train_file = open(train_file_name, "w")

    eval_file_name = os.path.join(data_dir, "edl_test.txt")
    eval_file = open(eval_file_name, "w")
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if i % 20 == 0:
                eval_file.write(line)
            else:
                train_file.write(line)

    train_file.close()
    eval_file.close()
    return train_file_name, eval_file_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records_per_shard",
        type=int,
        default=128,
        help="Record number per shard",
    )
    parser.add_argument(
        "--output_dir", help="The directory for the generated recordio files"
    )
    parser.add_argument(
        "--data_path", help="The path of the origin data file"
    )

    args = parser.parse_args(sys.argv[1:])

    train, val = split_to_train_test(args.data_path)

    convert_to_recordio_files(
        train, os.path.join(args.output_dir, "train"), args.records_per_shard
    )
    convert_to_recordio_files(
        val, os.path.join(args.output_dir, "val"), args.records_per_shard
    )
