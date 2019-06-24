import tensorflow as tf


class TFExampleCodec(object):
    def encode(self, example, feature_name_2_type):
        f_dict = {}
        for f_name, f_value in example.items():
            f_type = feature_name_2_type[f_name]
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

    def decode(self, raw, example_spec):
        return tf.io.parse_single_example(raw, example_spec)


codec = TFExampleCodec()
