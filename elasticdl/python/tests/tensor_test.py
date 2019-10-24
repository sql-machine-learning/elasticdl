"""Unittests for ElasticDL's Tensor data structure."""
import unittest

import numpy as np

from elasticdl.proto import elasticdl_pb2, tensor_dtype_pb2
from elasticdl.python.common.tensor import (
    Tensor,
    deserialize_tensor_pb,
    serialize_tensor,
)


class TensorTest(unittest.TestCase):
    def test_tensor_data_structure(self):
        # Test tensor values, without indices
        arr = np.ndarray(shape=[3, 1, 2, 4], dtype=np.int32)
        tensor = Tensor(arr)
        self.assertTrue(np.array_equal(arr, tensor.values))
        self.assertTrue(np.array_equal(arr, tensor.to_tf_tensor()))
        self.assertFalse(tensor.is_indexed_slices())

        # Test tensor values, with indices
        indices = np.array([2, 0, 1])
        tensor = Tensor(arr, indices)
        self.assertTrue(np.array_equal(arr, tensor.values))
        self.assertTrue(np.array_equal(indices, tensor.indices))
        self.assertTrue(np.array_equal(arr, tensor.to_tf_tensor().values))
        self.assertTrue(np.array_equal(indices, tensor.to_tf_tensor().indices))
        self.assertTrue(tensor.is_indexed_slices())

        # Test round trip
        # tensor to tensor PB
        tensor = Tensor(arr, indices, name="test")
        pb = tensor.to_tensor_pb()
        self.assertEqual(pb.name, "test")
        self.assertEqual(pb.dim, [3, 1, 2, 4])
        self.assertEqual(pb.dtype, tensor_dtype_pb2.DT_INT32)
        np.testing.assert_array_equal(pb.indices, indices)

        # tensor PB to tensor
        tensor_new = Tensor()
        tensor_new.from_tensor_pb(pb)
        self.assertEqual(tensor.name, "test")
        np.testing.assert_array_equal(tensor.values, arr)
        np.testing.assert_array_equal(tensor.indices, indices)

    def test_serialize_tensor(self):
        def _ndarray_to_tensor_pb(values, name=None, indices=None):
            tensor = Tensor(values, indices, name)
            tensor_pb = elasticdl_pb2.Tensor()
            serialize_tensor(tensor, tensor_pb)
            return tensor_pb

        # Wrong type, should raise
        arr = np.array([1, 2, 3, 4], dtype=np.uint8)
        with self.assertRaises(ValueError):
            _ndarray_to_tensor_pb(arr)

        # Empty array
        arr = np.array([], dtype=np.float32)
        t = _ndarray_to_tensor_pb(arr)
        self.assertEqual([0], t.dim)
        self.assertEqual(0, len(t.content))

        # Pathological case, one of the dimensions is 0.
        arr = np.ndarray(shape=[2, 0, 1, 9], dtype=np.float32)
        t = _ndarray_to_tensor_pb(arr)
        self.assertEqual([2, 0, 1, 9], t.dim)
        self.assertEqual(0, len(t.content))

        # 1-D array
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        t = _ndarray_to_tensor_pb(arr)
        self.assertEqual([4], t.dim)
        self.assertEqual(4 * 4, len(t.content))

        # 4-D random array
        arr = np.ndarray(shape=[2, 1, 3, 4], dtype=np.float32)
        t = _ndarray_to_tensor_pb(arr)
        self.assertEqual([2, 1, 3, 4], t.dim)
        self.assertEqual(4 * 2 * 1 * 3 * 4, len(t.content))

        # test name argument
        arr = np.ndarray(shape=[2, 1, 3, 4], dtype=np.float32)
        t = _ndarray_to_tensor_pb(arr, "test")
        self.assertTrue(t.name == "test")
        self.assertEqual([2, 1, 3, 4], t.dim)
        self.assertEqual(4 * 2 * 1 * 3 * 4, len(t.content))

        # test tf.IndexedSlices
        arr = np.ndarray(shape=[2, 1, 3, 4], dtype=np.float32)
        indices = np.array([3, 0], dtype=np.int64)
        t = _ndarray_to_tensor_pb(arr, "test", indices)
        self.assertTrue(t.name == "test")
        self.assertEqual([2, 1, 3, 4], t.dim)
        self.assertEqual([3, 0], t.indices)
        self.assertEqual(4 * 2 * 1 * 3 * 4, len(t.content))

    def test_deserialize_tensor_pb(self):
        pb = elasticdl_pb2.Tensor()
        tensor = Tensor()
        # No dim defined, should raise.
        self.assertRaises(ValueError, deserialize_tensor_pb, pb, tensor)

        # Empty array, should be ok.
        pb.dim.append(0)
        pb.content = b""
        pb.dtype = tensor_dtype_pb2.DT_FLOAT32
        deserialize_tensor_pb(pb, tensor)
        np.testing.assert_array_equal(
            np.array([], dtype=np.float32), tensor.values
        )

        # Wrong type, should raise
        del pb.dim[:]
        pb.dim.append(0)
        pb.content = b""
        pb.dtype = tensor_dtype_pb2.DT_INVALID
        self.assertRaises(ValueError, deserialize_tensor_pb, pb, tensor)

        # Pathological case, one of the dimensions is 0.
        del pb.dim[:]
        pb.dim.extend([2, 0, 1, 9])
        pb.content = b""
        pb.dtype = tensor_dtype_pb2.DT_FLOAT32
        deserialize_tensor_pb(pb, tensor)
        np.testing.assert_array_equal(
            np.ndarray(shape=[2, 0, 1, 9], dtype=np.float32), tensor.values
        )

        # Wrong content size, should raise
        del pb.dim[:]
        pb.dim.append(11)
        pb.content = b"\0" * (4 * 12)
        pb.dtype = tensor_dtype_pb2.DT_FLOAT32
        self.assertRaises(ValueError, deserialize_tensor_pb, pb, tensor)

        # Compatible dimensions, should be ok.
        for m in (1, 2, 3, 4, 6, 12):
            for with_inidices in [True, False]:
                del pb.dim[:]
                pb.content = b"\0" * (4 * 12)
                pb.dim.extend([m, 12 // m])
                if with_inidices:
                    pb.indices.extend([0] * m)
                pb.dtype = tensor_dtype_pb2.DT_FLOAT32
                deserialize_tensor_pb(pb, tensor)

    def testRoundTrip(self):
        def verify(values, name=None, indices=None):
            tensor = Tensor(values, indices, name)
            pb = elasticdl_pb2.Tensor()
            serialize_tensor(tensor, pb)
            deserialize_tensor_pb(pb, tensor)
            np.testing.assert_array_equal(values, tensor.values)
            if indices is not None:
                np.testing.assert_array_equal(indices, tensor.indices)
            if name:
                assert name == tensor.name

        # dtype = np.float32
        # 1-D array
        verify(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        # 4-D random array
        verify(np.ndarray(shape=[2, 1, 3, 4], dtype=np.float32))

        # verify with name
        verify(np.array([1, 2, 3, 4], dtype=np.float32), "test")

        # verify with indices
        verify(
            np.ndarray([2, 1, 3, 4], dtype=np.float32),
            "test",
            np.array([1, 0]),
        )

        # dtype = np.int64
        # 1-D random array
        verify(np.array([1, 2, 3, 4], dtype=np.int64))
        # 4-D random array
        verify(np.ndarray(shape=[2, 1, 3, 4], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
