import unittest
import tensorflow as tf
import numpy as np
from ps import ParameterServer


class ParameterServerTestCase(unittest.TestCase):
    def setUp(self):
        self.ps1 = ParameterServer(tf.train.GradientDescentOptimizer(0.1),
            {'x1': np.array([0.0, 0.0], dtype='float32'),
            'y1': np.array([1.0, 1.0], dtype='float32')})
        self.ps2 = ParameterServer(tf.train.GradientDescentOptimizer(0.1),
            {'x2': np.array([0.0, 0.0], dtype='float32'),
            'y2': np.array([1.0, 1.0], dtype='float32')})
        self.ps1.start()
        self.ps2.start()

    def tearDown(self):
        self.ps1.join()
        self.ps2.join()

    def test_all(self):
        # First pull before any update.
        step, params = self.ps1.pull()
        self.assertEqual(step, 0)
        np.testing.assert_array_almost_equal(
            params['x1'], np.array([0.0, 0.0], dtype='float32'))
        np.testing.assert_array_almost_equal(
            params['y1'], np.array([1.0, 1.0], dtype='float32'))

        self.ps1.push(0, 0,
            {'x1': np.array([0.1, 0.1], dtype='float32'),
            'y1': np.array([0.1, 0.1], dtype='float32')})

        # Pull the new version
        while step < 1:
            step, params = self.ps1.pull()
        self.assertEqual(step, 1)
        np.testing.assert_array_almost_equal(
            params['x1'], np.array([-0.01, -0.01], dtype='float32'))
        np.testing.assert_array_almost_equal(
            params['y1'], np.array([0.99, 0.99], dtype='float32'))

        # Pull a version in future should raise exception
        with self.assertRaises(LookupError):
            self.ps1.pull(min_step=2)
        
        # Pull by names
        step, params = self.ps1.pull(names=['x1',])
        self.assertEqual(step, 1)
        np.testing.assert_array_almost_equal(
            params['x1'], np.array([-0.01, -0.01], dtype='float32'))
        
        # Pull by unknown name
        with self.assertRaises(LookupError):
            self.ps1.pull(names=['z1',])



if __name__ == '__main__':
    unittest.main()
