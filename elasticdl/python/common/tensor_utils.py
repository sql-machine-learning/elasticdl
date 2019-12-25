import tensorflow as tf


def merge_indexed_slices(*args):
    return tf.IndexedSlices(
        tf.concat([i.values for i in args], axis=0),
        tf.concat([i.indices for i in args], axis=0),
    )


def deduplicate_indexed_slices(values, indices):
    """Sums `values` associated with any non-unique `indices`.
    Args:
        values: A `Tensor` with rank >= 1.
        indices: A one-dimensional integer `Tensor`, indexing into the first
        dimension of `values` (as in an IndexedSlices object).
    Returns:
        A tuple of (`summed_values`, `unique_indices`) where `unique_indices`
        is a de-duplicated version of `indices` and `summed_values` contains
        the sum of `values` slices associated with each unique index.
    """
    unique_indices, new_index_positions = tf.unique(indices)
    summed_values = tf.unsorted_segment_sum(
        values, new_index_positions, tf.shape(unique_indices)[0]
    )
    return (summed_values, unique_indices)
