import tensorflow as tf

class TFExampleBaseCodec(object):

    def __init__(self, feature_schema):
        self._f_schema = feature_schema
        self._f_desc = {}
        self._f_name2type = {}
        for f_name, f_type, in self._f_schema:
            self._f_name2type[f_name] = f_type
            if f_type == tf.string:
                self._f_desc[f_name] = tf.FixedLenFeature([], tf.string, default_value='')
            elif f_type in (tf.float32, tf.float64):
                self._f_desc[f_name] = tf.FixedLenFeature([], f_type, default_value=0.0)
            elif f_type in (tf.bool, tf.int32, tf.uint32, tf.int64, tf.uint64):
                self._f_desc[f_name] = tf.FixedLenFeature([], f_type, default_value=0)
            else:
                raise ValueError("not supported tensorflow data type.")

class TFExampleEncoder(TFExampleBaseCodec):

    def __call__(self, example):
        f_dict = {}
        for f_name, f_value in example:
            f_type = self._f_name2type[f_name]
            if f_type == tf.string:
                f_dict[f_name] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[f_value]))
            elif f_type in (tf.float32, tf.float64):
                f_dict[f_name] = tf.train.Feature(float_list=tf.train.FloatList(value=[f_value]))
            elif f_type in (tf.bool, tf.int32, tf.uint32, tf.int64, tf.uint64):
                f_dict[f_name] = tf.train.Feature(int64_list=tf.train.Int64List(value=[f_value]))
            else:
                raise ValueError("not supported tensorflow data type.")

        example = tf.train.Example(features=tf.train.Features(feature=f_dict))
        return example.SerializeToString()

class TFExampleDecoder(TFExampleBaseCodec):

    def __call__(self, example):
        return tf.parse_single_example(example, self._f_desc)
