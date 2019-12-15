import argparse
import os
import pathlib
import sys
import urllib

import pandas as pd
import recordio
import tensorflow as tf
from sklearn.model_selection import train_test_split

TRAIN_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    "adult.data"
)
TEST_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    "adult.test"
)

__COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "label",
]
CATEGORICAL_FEATURE_KEYS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
NUMERIC_FEATURE_KEYS = [
    "age",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
OPTIONAL_NUMERIC_FEATURE_KEYS = [
    "education-num",
]
LABEL_KEY = "label"


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
    if data_series.hasnans:
        return

    features = {}
    for numeric_feature_key in NUMERIC_FEATURE_KEYS:
        feature = tf.train.Feature(
            float_list=tf.train.FloatList(
                value=[data_series[numeric_feature_key]]
            )
        )
        features[numeric_feature_key] = feature

    for categorical_feature_key in CATEGORICAL_FEATURE_KEYS:
        feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[data_series[categorical_feature_key].encode("utf-8")]
            )
        )
        features[categorical_feature_key] = feature

    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[data_series[LABEL_KEY]])
    )
    features[LABEL_KEY] = feature

    return features


def convert_to_recordio_files(data_frame, dir_name, records_per_shard):
    """
    Convert a pandas DataFrame to recordio files.
    Args:
        data_frame: A pandas DataFrame to convert_to_recordio_files.
        dir_name: A directory to put the generated recordio files.
        records_per_shard: The record number per shard.
    """
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)

    row_num = 0
    writer = None
    for index, row in data_frame.iterrows():
        if row_num % records_per_shard == 0:
            if writer:
                writer.close()

            shard = row_num // records_per_shard
            file_path_name = os.path.join(dir_name, "data-%05d" % shard)
            writer = recordio.Writer(file_path_name)

        feature = convert_series_to_tf_feature(
            row, data_frame.columns, data_frame.dtypes
        )
        result_string = tf.train.Example(
            features=tf.train.Features(feature=feature)
        ).SerializeToString()
        writer.write(result_string)

        row_num += 1

    if writer:
        writer.close()

    print("Finish data conversion in {}".format(dir_name))


def load_raw_data(source_data_url, data_dir):
    file_name = os.path.basename(source_data_url)
    file_path = os.path.join(data_dir, file_name)
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(source_data_url, file_path)
    census = pd.read_csv(file_path, skiprows=1, header=None, skipinitialspace=True)
    census.columns = __COLUMN_NAMES

    census[LABEL_KEY] = census[LABEL_KEY].apply(
        lambda label: 0 if label == "<=50K" else 1
    )
    for optional_key in OPTIONAL_NUMERIC_FEATURE_KEYS:
        census.pop(optional_key)

    return census


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        help="The cache directory to put the data downloaded from the web",
        required=True
    )
    parser.add_argument(
        "--records_per_shard",
        type=int,
        default=1024 * 8,
        help="Record number per shard",
    )
    parser.add_argument(
        "--output_dir",
        help="The directory for the generated recordio files",
        required=True
    )

    args = parser.parse_args(sys.argv[1:])

    train_data_frame = load_raw_data(TRAIN_DATA_URL, args.data_dir)
    test_data_frame = load_raw_data(TEST_DATA_URL, args.data_dir)

    convert_to_recordio_files(
        train_data_frame, os.path.join(args.output_dir, "train"), args.records_per_shard
    )
    convert_to_recordio_files(
        test_data_frame, os.path.join(args.output_dir, "test"), args.records_per_shard
    )
