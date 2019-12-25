import tensorflow as tf


def merge_indexed_slices(*args):
    return tf.IndexedSlices(
        tf.concat([i.values for i in args], axis=0),
        tf.concat([i.indices for i in args], axis=0),
    )


def deduplicate_indexed_slices(values, indices):
    """
    Sum up the values associated with duplicated indices in the IndexedSlices.
    Args:
        values: A Tensor with rank >= 1. Particularly IndexedSlices.values.
        indices: A one-dimension integer of Tensor. Particularly 
        IndexedSlices.indices.
    Returns:
        A tuple of (`sum_combined_values`, `unique_indices`).
        `sum_combined_values` contains the sum of `values` associated
        with each unique indice.
        `unique_indices` is a de-duplicated version of `indices`.
    """
    unique_indices, new_index_positions = tf.unique(indices)
    sum_combined_values = tf.math.unsorted_segment_sum(
        values, new_index_positions, tf.shape(unique_indices)[0]
    )

    return (sum_combined_values, unique_indices)
