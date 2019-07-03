import tensorflow as tf


def prepare_data_for_a_single_file(file_object, filename):
    """
    :param filename: training data file name
    The raw images should be packed into a TAR file and named as
    <label_id>_xxx.JPEG, in which label_id is an integer representing
    the category the image belongs to, and xxx can be anything.

    :param file_object: a file object associated with filename
    """
    label = filename.split("/")[-1].split("_")[0]
    int_label = int(label)
    image_bytes = file_object.read()
    feature_dict = {}
    feature_dict["image"] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[image_bytes])
    )
    feature_dict["label"] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[int_label])
    )
    example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict)
    )
    return example.SerializeToString()
