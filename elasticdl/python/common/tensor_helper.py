import tensorflow as tf


def merge_indexed_slices(*args):
    return tf.IndexedSlices(
        tf.concat([i.values for i in args], axis=0),
        tf.concat([i.indices for i in args], axis=0),
    )
