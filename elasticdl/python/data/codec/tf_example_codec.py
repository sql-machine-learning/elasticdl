import tensorflow as tf


class TFExampleCodec(object):
    def __init__(self, feature_columns):
        self._example_spec = tf.feature_column.make_parse_example_spec(
            feature_columns
        )
        self._f_name2type = {
            f_col.key: f_col.dtype for f_col in feature_columns
        }

    def encode(self, example):
        if self._example_spec.keys() != example.keys():
            raise ValueError(
                "Column keys mismatch: expected %s, got %s "
                % (self._example_spec.keys(), example.keys())
            )
        f_dict = {}
        for f_name, f_value in example.items():
            f_type = self._f_name2type[f_name]
            if f_type == tf.string:
                f_dict[f_name] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=f_value)
                )
            elif f_type == tf.float32:
                f_dict[f_name] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=f_value.flatten())
                )
            elif f_type == tf.int64:
                f_dict[f_name] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=f_value.flatten())
                )
            else:
                raise ValueError(
                    "not supported tensorflow data type: " + str(f_type)
                )

        example = tf.train.Example(features=tf.train.Features(feature=f_dict))
        return example.SerializeToString()

    def decode(self, raw):
        return tf.io.parse_single_example(raw, self._example_spec)
