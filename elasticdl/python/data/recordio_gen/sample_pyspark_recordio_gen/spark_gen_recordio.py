import argparse
import os
import glob
import tarfile
from pyspark import SparkContext
from pyspark import TaskContext

from elasticdl.python.common.model_helper import load_module
from contextlib import closing
import recordio


def write_to_recordio(filename, data_list):
    print("Writing to file:", filename)
    with closing(recordio.Writer(filename)) as f:
        for d in data_list:
            f.write(d)


def process_data(
    single_file_preparation_func,
    training_data_tar_file,
    output_dir,
    records_per_file,
):
    def _process_data(filename_list):
        filename_set = set()
        for filename in filename_list:
            filename_set.add(filename)

        tar = tarfile.open(training_data_tar_file)
        tar_info_list = tar.getmembers()
        filename_to_object = {}
        for tar_info in tar_info_list:
            if tar_info.name in filename_set:
                f = tar.extractfile(tar_info)
                assert f is not None
                filename_to_object[tar_info.name] = f

        partition = TaskContext().partitionId()
        counter = 0
        data_list = []
        for filename in glob.glob(output_dir + "/data-%s*" % partition):
            os.remove(filename)
        for filename in filename_set:
            data = single_file_preparation_func(
                filename_to_object[filename], filename
            )
            data_list.append(data)
            if len(data_list) == records_per_file:
                filename = output_dir + "/data-%s-%04d" % (partition, counter)
                counter += 1

                write_to_recordio(filename, data_list)
                data_list.clear()

        if data_list:
            filename = output_dir + "/data-%s-%04d" % (partition, counter)
            write_to_recordio(filename, data_list)
        return filename_list

    return _process_data


def main():
    parser = argparse.ArgumentParser(
        description="Spark job to convert training data to RecordIO format"
    )
    parser.add_argument(
        "--training_data_tar_file",
        help="Tar file that contains all training data",
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
        "--num_workers",
        default=2,
        type=int,
        help="Number of workers of Spark job",
    )

    args = parser.parse_args()

    # Get training data file names from training_data_tar_file
    tar = tarfile.open(args.training_data_tar_file)
    tar_info_list = tar.getmembers()
    filename_list = []
    for tar_info in tar_info_list:
        f = tar.extractfile(tar_info)
        if f is not None and not tar_info.name.split("/")[-1].startswith("."):
            filename_list.append(tar_info.name)

    # Load user-defined model
    model_module = load_module(args.model_file)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Start the Spark job
    sc = SparkContext()
    rdd = sc.parallelize(filename_list, args.num_workers)
    rdd.mapPartitions(
        process_data(
            model_module.prepare_data_for_a_single_file,
            args.training_data_tar_file,
            args.output_dir,
            args.records_per_file,
        )
    ).collect()


if __name__ == "__main__":
    main()
