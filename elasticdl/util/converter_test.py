import numpy as np
import unittest

from proto import master_pb2
from .converter import NdarrayToTensor, TensorToNdarray


class ConverterTest(unittest.TestCase):
    def testNdarrayToTensor(self):
        # Wrong type, should raise
        arr = np.array([1, 2, 3, 4])
        self.assertRaises(ValueError, NdarrayToTensor, arr)

        # 1-D array
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        t = NdarrayToTensor(arr)
        self.assertEqual([4], t.dim)
        self.assertEqual(4 * 4, len(t.content))

        # 4-D random array
        arr = np.ndarray(shape=[2, 1, 3, 4], dtype=np.float32)
        t = NdarrayToTensor(arr)
        self.assertEqual([2, 1, 3, 4], t.dim)
        self.assertEqual(4 * 2 * 1 * 3 * 4, len(t.content))

    def testTensorToNdarray(self):
        t = master_pb2.Tensor()
        t.content = b'\0' * (4 * 12)
        
        # Wrong content size, should raise
        t.dim.extend([11,])
        self.assertRaises(ValueError, TensorToNdarray, t)

        # Compatible dimensions, should be ok.
        for m in (1, 2, 3, 4, 6, 12):
            del t.dim[:]
            t.content = b'\0' * (4 * 12)
            t.dim.extend([m, 12 // m])
            arr = TensorToNdarray(t)

    def testRoundTrip(self):
        def verify(a):
            b = TensorToNdarray(NdarrayToTensor(a))
            np.testing.assert_array_equal(a, b)

        # 1-D array
        verify(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        # 4-D random array
        verify(np.ndarray(shape=[2, 1, 3, 4], dtype=np.float32))
