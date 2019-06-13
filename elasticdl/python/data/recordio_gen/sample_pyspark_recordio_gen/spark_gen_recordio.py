import argparse
import os

from pyspark import SparkContext
from pyspark import TaskContext

import numpy as np
from elasticdl.python.elasticdl.common.model_helper import load_user_model
from elasticdl.python.data.recordio_gen.convert_numpy_to_recordio import (
    convert_numpy_to_recordio,
)


def process_data(
    feature_label_columns,
    single_file_preparation_func,
    output_dir,
    records_per_file,
    codec_type,
):
    def _process_data(file_list):
        ctx = TaskContext()
        feature_list = []
        label_list = []
        for file in file_list:
            feature_label_tuple = single_file_preparation_func(file)
            assert len(feature_label_tuple) == 2
            feature_list.append(feature_label_tuple[0])
            label_list.append(feature_label_tuple[1])
        convert_numpy_to_recordio(
            output_dir,
            np.array(feature_list),
            np.array(label_list),
            feature_label_columns,
            records_per_file,
            codec_type,
            str(ctx.partitionId()),
        )
        return file_list

    return _process_data


def main():
    parser = argparse.ArgumentParser(
        description="Spark job to convert training data to RecordIO format"
    )
    parser.add_argument(
        "--training_data_dir",
        help="Directory that contains training data and will be traversed \
            recursively",
        required=True,
    )
    parser.add_argument(
        "--output_dir", help="Directory of output RecordIO data", required=True
    )
    parser.add_argument(
        "--model_file",
        required=True,
        help="User-defined model file which data processing logic is in",
    )
    parser.add_argument(
        "--records_per_file", default=1024, type=int, help="Record per file"
    )
    parser.add_argument(
        "--codec_type",
        default="tf_example",
        choices=["tf_example", "bytes"],
        help="Type of codec(tf_example or bytes)",
    )
    parser.add_argument(
        "--num_workers",
        default=2,
        type=int,
        help="Number of workers of Spark job",
    )

    args = parser.parse_args()

    # Get training data files recursively from training_data_dir
    file_list = []
    for dir_name, subdir_list, files in os.walk(args.training_data_dir):
        for fname in files:
            if not fname.startswith("."):
                file_list.append(os.path.join(dir_name, fname))

    # Load user-defined model
    model_module = load_user_model(args.model_file)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Start the Spark job
    sc = SparkContext()
    rdd = sc.parallelize(file_list, args.num_workers)
    rdd.mapPartitions(
        process_data(
            model_module.feature_columns() + model_module.label_columns(),
            model_module.prepare_data_for_a_single_file,
            args.output_dir,
            args.records_per_file,
            args.codec_type,
        )
    ).collect()


if __name__ == "__main__":
    main()
