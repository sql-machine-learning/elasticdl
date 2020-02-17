import unittest

import numpy as np
import tensorflow as tf

from elasticdl.python.common.tensor_utils import (
    indexed_slices_to_pb,
    ndarray_to_pb,
    pb_to_indexed_slices,
    pb_to_ndarray,
)


class TensorUtilsTest(unittest.TestCase):
    def test_round_trip(self):
        def verify(array):
            pb = ndarray_to_pb(array)
            new_array = pb_to_ndarray(pb)
            np.testing.assert_array_equal(array, new_array)

        # dtype = np.float32
        # 1-D array
        verify(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        # 4-D random array
        verify(np.ndarray(shape=[2, 1, 3, 4], dtype=np.float32))

        # dtype = np.int64
        # 1-D random array
        verify(np.array([1, 2, 3, 4], dtype=np.int64))
        # 4-D random array
        verify(np.ndarray(shape=[2, 1, 3, 4], dtype=np.int64))

    def test_indexed_slices_round_trip(self):
        def verify(slices):
            pb = indexed_slices_to_pb(slices)
            new_slices = pb_to_indexed_slices(pb)
            np.testing.assert_array_equal(slices.values, new_slices.values)
            np.testing.assert_array_equal(slices.indices, new_slices.indices)

        # dtype = np.float32
        verify(
            tf.IndexedSlices(
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.array([0, 2, 1]),
            )
        )
        # dtype = np.int64
        verify(
            tf.IndexedSlices(
                np.array([1, 2, 3], dtype=np.int64), np.array([0, 2, 1])
            )
        )

        slices = tf.IndexedSlices(
            np.array([1, 2, 3], dtype=np.int64),
            np.array([[0, 1], [1, 2], [2, 3]]),
        )
        self.assertRaisesRegex(
            ValueError,
            "IndexedSlices pb only accepts indices with one dimension",
            verify,
            slices,
        )


if __name__ == "__main__":
    unittest.main()
