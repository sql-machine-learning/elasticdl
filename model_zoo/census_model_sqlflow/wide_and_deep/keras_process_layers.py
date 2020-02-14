import tensorflow as tf
from tensorflow.python.ops import lookup_ops, math_ops


class CategoryHash(tf.keras.layers.Layer):
    """
    Todo replace it with a preprocess layer after TF 2.2
    https://github.com/tensorflow/community/pull/188/files?short_path=0657914#diff-0657914a8dc40e5fbca67680bf3fc45f
    """

    def __init__(self, bucket_size):
        super(CategoryHash, self).__init__()
        self.bucket_size = bucket_size

    def call(self, inputs):
        if inputs.dtype is not tf.string:
            inputs = tf.strings.as_string(inputs)
        bucket_id = tf.strings.to_hash_bucket_fast(inputs, self.bucket_size)
        return tf.cast(bucket_id, tf.int64)


class NumericBucket(tf.keras.layers.Layer):
    def __init__(self, boundaries):
        super(NumericBucket, self).__init__()
        self.boundaries = boundaries

    def call(self, inputs):
        if inputs.dtype is tf.string:
            inputs = tf.strings.to_number(inputs, out_type=tf.float32)
        else:
            inputs = tf.cast(inputs, tf.float32)
        bucket_id = math_ops._bucketize(inputs, boundaries=self.boundaries)
        return tf.cast(bucket_id, tf.int64)


class CategoryLookup(tf.keras.layers.Layer):
    """
    Todo replace it with a preprocess layer after TF 2.2
    https://github.com/tensorflow/community/pull/188/files?short_path=0657914#diff-0657914a8dc40e5fbca67680bf3fc45f
    """

    def __init__(self, vocabulary_list, num_oov_buckets=1, default_value=-1):
        super(CategoryLookup, self).__init__()
        self.vocabulary_list = vocabulary_list

    def call(self, inputs):
        table = lookup_ops.index_table_from_tensor(
            vocabulary_list=self.vocabulary_list,
            num_oov_buckets=1,
            default_value=-1,
        )
        return tf.cast(table.lookup(inputs), tf.int64)


class Group(tf.keras.layers.Layer):
    def __init__(self, offsets):
        super(Group, self).__init__()
        self.offsets = offsets

    def call(self, inputs):
        if self.offsets is None:
            return tf.keras.backend.stack(inputs, axis=1)

        ids_with_offset = []
        if len(self.offsets) != len(inputs):
            raise ValueError(
                "The number of elements in offsets is not equal to inputs"
            )
        for i, value in enumerate(inputs):
            ids_with_offset.append(value + self.offsets[i])

        return tf.keras.backend.stack(ids_with_offset, axis=1)
