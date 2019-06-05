import argparse
import os
import tensorflow as tf
from pyspark import SparkContext
from pyspark import TaskContext

import PIL.Image
import numpy as np
from elasticdl.python.data.recordio_gen.convert_numpy_to_recordio import \
    convert_numpy_to_recordio


def process_data(image_shape, output_dir, records_per_file, codec_type):
    def _process_data(file_list):
        ctx = TaskContext()
        numpy_image_list = []
        label_list = []
        for file in file_list:
            label_list.append(int(file.split('/')[-2]))
            image = PIL.Image.open(file)
            numpy_image = np.array(image)
            assert image_shape == list(numpy_image.shape)
            numpy_image_list.append(numpy_image)

        image_numpy_array = np.array(numpy_image_list)
        label_array = np.array(label_list)
        feature_columns = [
            tf.feature_column.numeric_column(
                key="image", dtype=tf.float32, shape=image_shape
            ),
            tf.feature_column.numeric_column(
                key="label", dtype=tf.int64, shape=[1]
            ),
        ]
        convert_numpy_to_recordio(
            output_dir,
            image_numpy_array,
            label_array,
            feature_columns,
            records_per_file=records_per_file,
            codec_type=codec_type,
            partition=str(ctx.partitionId()),
        )
        return file_list
    return _process_data


def get_image_shape(file):
    image = PIL.Image.open(file)
    numpy_image = np.array(image)
    return list(numpy_image.shape)


def main():
    parser = argparse.ArgumentParser(
        description="Spark job to generate image classfication task "
                    "training/eval data in RecordIO format. All images should "
                    "be the same size and located in the sub directories that "
                    "are named by the corresponding catetory number, which "
                    "starts from 0 and increases monotonously. e.g:"
                    "training_data_dir/0/image1, training_data_dir/0/image2,"
                    "training_data_dir/1/image3, ..."
    )
    parser.add_argument(
        "--training_data_dir",
        help="Dir that contains training data organized by classes",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="Dir of output RecordIO data",
        required=True,
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
        help="Number of workers",
    )

    # TODO: Add cluster mode
    parser.add_argument(
        "--mode",
        choices=["local", "cluster"],
        default='local',
        help="Spark job mode",
    )

    args = parser.parse_args()

    file_list = []
    for dir_name, subdir_list, files in os.walk(args.training_data_dir):
        for fname in files:
            if fname == '.DS_Store':
                continue
            file_list.append(os.path.join(dir_name, fname))

    image_shape = get_image_shape(file_list[0])
    print('image shape is:', image_shape)

    sc = SparkContext(args.mode)
    rdd1 = sc.parallelize(file_list, args.num_workers)
    rdd1.mapPartitions(
        process_data(
            image_shape,
            args.output_dir,
            args.records_per_file,
            args.codec_type,
        )
    ).collect()

if __name__ == "__main__":
    main()
