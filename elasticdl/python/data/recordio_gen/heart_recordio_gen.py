import argparse
import pandas as pd 
import tensorflow as tf 
import recordio
import os
import pathlib
import urllib
import sys
from sklearn.model_selection import train_test_split

URL = 'https://storage.googleapis.com/applied-dl/heart.csv'

def convert_series_to_tf_feature(data_series, columns, dtype_series):
    features = {}
    for column_name in columns:
        feature = None
        value = data_series[column_name]
        dtype = dtype_series[column_name]

        if dtype == 'int64':
            feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        elif dtype == 'float64':
            feature = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        elif dtype == 'str':
            feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))
        elif dtype == 'object':
            feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value).encode('utf-8')]))
        else:
            assert False, 'Unrecoginize dtype: {}'.format(dtype)

        if feature:
            features[column_name] = feature

    return features

def convert(data_frame, root_dir, sub_dir, records_per_shard):
    dir_name = os.path.join(root_dir, sub_dir)
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)

    row_num = 0
    writer = None
    for index, row in data_frame.iterrows():
        if row_num % records_per_shard == 0:
            if writer:
                writer.close()

            shard = row_num // records_per_shard
            file_path_name = os.path.join(dir_name, 'data-%05d' % shard)
            writer = recordio.Writer(file_path_name)

        feature = convert_series_to_tf_feature(row, data_frame.columns, data_frame.dtypes)
        result_string = tf.train.Example(
                features=tf.train.Features(
                    feature=feature
                )
            ).SerializeToString()
        writer.write(result_string)

        row_num += 1

    if writer:
        writer.close()

    print('Finish data conversion in {}'.format(dir_name))

def load_raw_data(data_dir):
    file_name = os.path.basename(URL)
    file_path = os.path.join(data_dir, file_name)
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(URL, file_path)
    return pd.read_csv(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='The cache directory to put the downloaded data from the web')
    parser.add_argument('--records_per_shard', type=int, default=128, help='Record number per shard')
    parser.add_argument('--output_dir', help='The directory for the generated recordio files')

    args = parser.parse_args(sys.argv[1:])

    data_frame = load_raw_data(args.data_dir)

    train, test = train_test_split(data_frame, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    convert(train, args.output_dir, 'train', args.records_per_shard)
    convert(val, args.output_dir, 'val', args.records_per_shard)
    convert(test, args.output_dir, 'test', args.records_per_shard)
