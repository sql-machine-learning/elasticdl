# Copyright 2020 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import pathlib
import sys

import recordio
import tensorflow as tf

COLUMNS = (
    ["label"]
    + ["I" + str(i) for i in range(1, 14)]
    + ["C" + str(i) for i in range(1, 27)]
)


def convert_data_to_tf_example(sample_data, columns):
    features = {}
    column_data = sample_data.split("\t")
    for i, column_name in enumerate(columns):
        value = column_data[i].strip()
        if column_name[0] == "I" or column_name == "label":
            value = 0 if value == "" else int(value)
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[value])
            )
        else:
            feature = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[value.encode("utf-8")])
            )
        features[column_name] = feature

    example = tf.train.Example(
        features=tf.train.Features(feature=features)
    ).SerializeToString()

    return example


def convert_to_recordio_files(file_path, dir_name, records_per_shard=10240):
    """
    Convert a CSV file to recordio files.
    Args:
        file_path: A path of the CSV file
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
            example = convert_data_to_tf_example(row, COLUMNS)
            writer.write(example)

        if writer:
            writer.close()


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
    parser.add_argument("--data_path", help="The path of the origin data file")

    args = parser.parse_args(sys.argv[1:])

    train, val = split_to_train_test(args.data_path)

    convert_to_recordio_files(
        train, os.path.join(args.output_dir, "train"), args.records_per_shard
    )
    convert_to_recordio_files(
        val, os.path.join(args.output_dir, "val"), args.records_per_shard
    )
