"""Unittests for ElasticDL's Tensor data structure."""
import unittest

import numpy as np

from elasticdl.proto import elasticdl_pb2, tensor_dtype_pb2
from elasticdl.python.common.dtypes import dtype_numpy_to_tensor
from elasticdl.python.common.tensor import (
    Tensor,
    deserialize_tensor_pb,
    emplace_tensor_pb_from_ndarray,
    serialize_tensor,
    tensor_pb_to_ndarray,
    tensor_pb_to_tf_tensor,
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
        tensor_new = Tensor.from_tensor_pb(pb)
        self.assertEqual(tensor.name, "test")
        np.testing.assert_array_equal(tensor_new.values, arr)
        np.testing.assert_array_equal(tensor_new.indices, indices)

        # Test Tensor().to_ndarray()
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        indices = np.array([0, 2])
        name = "test"
        tensor = Tensor(values, indices, name)
        self.assertRaises(NotImplementedError, tensor.to_ndarray)
        tensor = Tensor(values, name=name)
        self.assertTrue(np.allclose(values, tensor.to_ndarray()))

    def test_serialize_tensor(self):
        def _ndarray_to_tensor_pb(values, name=None, indices=None):
            return Tensor(values, indices, name).to_tensor_pb()

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
                self.assertEqual((m, 12 // m), tensor.values.shape)
                self.assertTrue(isinstance(tensor.values, np.ndarray))
                if tensor.indices is not None:
                    self.assertTrue(isinstance(tensor.indices, np.ndarray))

    def test_round_trip(self):
        def verify(values, name=None, indices=None):
            tensor = Tensor(values, indices, name)
            pb = elasticdl_pb2.Tensor()
            serialize_tensor(tensor, pb)
            tensor_new = Tensor()
            deserialize_tensor_pb(pb, tensor_new)
            np.testing.assert_array_equal(values, tensor_new.values)
            if indices is not None:
                np.testing.assert_array_equal(indices, tensor_new.indices)
            if name:
                self.assertEqual(name, tensor.name)

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

    def _create_tensor_pb(self, values, indices=None):
        pb = elasticdl_pb2.Tensor()
        pb.dim.extend(values.shape)
        pb.dtype = dtype_numpy_to_tensor(values.dtype)
        pb.content = values.tobytes()
        if indices is not None:
            pb.indices.extend(tuple(indices))
        return pb

    def test_tensor_pb_to_ndarray(self):
        values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], np.float32)
        indices = np.array([0, 2])
        pb = self._create_tensor_pb(values)
        self.assertTrue(np.allclose(tensor_pb_to_ndarray(pb), values))

        # convert a tensor_pb with sparse tensor to a ndarray, should raise
        pb = self._create_tensor_pb(values, indices)
        self.assertRaises(NotImplementedError, tensor_pb_to_ndarray, pb)

    def test_tensor_pb_to_tf_tensor(self):
        values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], np.float32)
        indices = np.array([0, 2])

        # Test dense tensor
        pb = self._create_tensor_pb(values)
        self.assertTrue(np.allclose(tensor_pb_to_tf_tensor(pb), values))

        # Test sparse tensor
        pb = self._create_tensor_pb(values, indices)
        tf_tensor = tensor_pb_to_tf_tensor(pb)
        self.assertTrue(np.allclose(tf_tensor.values, values))
        self.assertTrue(np.allclose(tf_tensor.indices, indices))

    def test_emplace_tensor_pb_from_ndarray(self):
        values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], np.float32)
        indices = np.array([0, 2])
        name = "test"
        model = elasticdl_pb2.Model()
        emplace_tensor_pb_from_ndarray(model.param, values, indices, name)
        pb = model.param[-1]
        print("pb", pb)

        expected_pb = Tensor(values, indices, name).to_tensor_pb()
        self.assertEqual(pb.name, expected_pb.name)
        self.assertEqual(pb.dim, expected_pb.dim)
        self.assertEqual(pb.content, expected_pb.content)
        self.assertEqual(pb.indices, expected_pb.indices)
        self.assertEqual(pb.dtype, expected_pb.dtype)


if __name__ == "__main__":
    unittest.main()
