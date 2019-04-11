from recordio import File
from recordio.recordio.header import Compressor
import tensorflow as tf
from reader_wrapper import RangeReaderWrapper

class TFExampleFile(object):
    """ A pyrecordio file where each single record is type of tf.train.Example. 
    """

    def __init__(self, file_path, feature_schemas, mode, *, max_chunk_size=1024, 
                 compressor=Compressor.snappy):
        self._rio_file = File(file_path, mode, max_chunk_size=max_chunk_size, compressor=compressor)
        self._f_schemas = feature_schemas
        self._f_desc = {}
        self._f_name2type = {}
        for f_schema in self._f_schemas:
            f_name = f_schema[0]
            f_type = f_schema[1]
            self._f_name2type[f_name] = f_type
            if f_type == tf.string:
                self._f_desc[f_name] = tf.FixedLenFeature([], tf.string, default_value='')
            elif f_type == tf.float32 or f_type == tf.float64:
                self._f_desc[f_name] = tf.FixedLenFeature([], f_type, default_value=0.0)
            elif f_type == tf.bool or f_type == tf.int32 or f_type == tf.uint32 \
                or f_type == tf.int64 or f_type == tf.uint64:
                self._f_desc[f_name] = tf.FixedLenFeature([], f_type, default_value=0)

    def __enter__(self):
        """ For `with` statement
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ For `with` statement
        """
        self._rio_file.close()

    def __iter__(self):
        """ For iterate operation
        Returns:
            Iterator of dataset
        """
        return RangeReaderWrapper(self._rio_file.get_reader(), self._f_desc)

    def get(self, index):
        """ Get a single tf.train.Example data specified by index
        Arguments:
            index: record index in the recordio file.
        Returns:
            A dict where the key is feature_name and value is feature_value. 
        Raises:
            RuntimeError: wrong open mode.
        """
        serialized_example = self._rio_file.get(index)
        return tf.parse_single_example(serialized_example, self._f_desc)

    def write(self, features):
        """ Write a single features row into recordio file.
        Arguments:
            features: a list where each item is a feature being a tuple which contains 
                      feature_name and feature_data.
        Raises:
            RuntimeError: wrong open mode.
        """
        f_dict = {}
        for feature in features:
            f_name = feature[0]
            f_value = feature[1]
            f_type = self._f_name2type[f_name]
            if f_type == tf.string:
                f_dict[f_name] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[f_value])) 
            elif f_type == tf.float32 or f_type == tf.float64:
                f_dict[f_name] = tf.train.Feature(float_list=tf.train.FloatList(value=[f_value])) 
            elif f_type == tf.bool or f_type == tf.int32 or f_type == tf.uint32 \
                or f_type == tf.int64 or f_type == tf.uint64:
                f_dict[f_name] = tf.train.Feature(int64_list=tf.train.Int64List(value=[f_value])) 

        example = tf.train.Example(features=tf.train.Features(feature=f_dict))
        serialized_example = example.SerializeToString()
        self._rio_file.write(serialized_example)

    def close(self):
        self._rio_file.close() 
