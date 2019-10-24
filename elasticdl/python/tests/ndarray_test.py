import unittest

import numpy as np

from elasticdl.proto import elasticdl_pb2, tensor_dtype_pb2
from elasticdl.python.common.ndarray import (
    ndarray_to_tensor,
    tensor_to_ndarray,
)


class ConverterTest(unittest.TestCase):
    def test_ndarray_to_tensor(self):
        # Wrong type, should raise
        arr = np.array([1, 2, 3, 4], dtype=np.uint8)
        self.assertRaises(ValueError, ndarray_to_tensor, arr)

        # Empty array
        arr = np.array([], dtype=np.float32)
        t = ndarray_to_tensor(arr)
        self.assertEqual([0], t.dim)
        self.assertEqual(0, len(t.content))

        # Pathological case, one of the dimensions is 0.
        arr = np.ndarray(shape=[2, 0, 1, 9], dtype=np.float32)
        t = ndarray_to_tensor(arr)
        self.assertEqual([2, 0, 1, 9], t.dim)
        self.assertEqual(0, len(t.content))

        # 1-D array
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        t = ndarray_to_tensor(arr)
        self.assertEqual([4], t.dim)
        self.assertEqual(4 * 4, len(t.content))

        # 4-D random array
        arr = np.ndarray(shape=[2, 1, 3, 4], dtype=np.float32)
        t = ndarray_to_tensor(arr)
        self.assertEqual([2, 1, 3, 4], t.dim)
        self.assertEqual(4 * 2 * 1 * 3 * 4, len(t.content))

        # test name argument
        arr = np.ndarray(shape=[2, 1, 3, 4], dtype=np.float32)
        t = ndarray_to_tensor(arr, name="test")
        self.assertTrue(t.name == "test")
        self.assertEqual([2, 1, 3, 4], t.dim)
        self.assertEqual(4 * 2 * 1 * 3 * 4, len(t.content))

    def testtensor_to_ndarray(self):
        t = elasticdl_pb2.Tensor()
        # No dim defined, should raise.
        self.assertRaises(ValueError, tensor_to_ndarray, t)

        # Empty array, should be ok.
        t.dim.append(0)
        t.content = b""
        t.dtype = tensor_dtype_pb2.DT_FLOAT32
        arr = tensor_to_ndarray(t)
        np.testing.assert_array_equal(np.array([], dtype=np.float32), arr)

        # Pathological case, one of the dimensions is 0.
        del t.dim[:]
        t.dim.extend([2, 0, 1, 9])
        t.content = b""
        t.dtype = tensor_dtype_pb2.DT_FLOAT32
        arr = tensor_to_ndarray(t)
        np.testing.assert_array_equal(
            np.ndarray(shape=[2, 0, 1, 9], dtype=np.float32), arr
        )

        t.content = b"\0" * (4 * 12)

        # Wrong content size, should raise
        del t.dim[:]
        t.dim.extend([11])
        t.dtype = tensor_dtype_pb2.DT_FLOAT32
        self.assertRaises(ValueError, tensor_to_ndarray, t)

        # Compatible dimensions, should be ok.
        for m in (1, 2, 3, 4, 6, 12):
            del t.dim[:]
            t.content = b"\0" * (4 * 12)
            t.dim.extend([m, 12 // m])
            t.dtype = tensor_dtype_pb2.DT_FLOAT32
            arr = tensor_to_ndarray(t)

    def testRoundTrip(self):
        def verify(a, name=None):
            b = tensor_to_ndarray(ndarray_to_tensor(a, name=name))
            if name:
                assert name == b.name
            np.testing.assert_array_equal(a, b)

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


if __name__ == "__main__":
    unittest.main()
