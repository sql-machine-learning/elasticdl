import unittest
import tensorflow as tf
import numpy as np
import threading
import time
from ps import ParameterServer
from timeit import default_timer as timer


class ParameterServerTestCase(unittest.TestCase):
    def setUp(self):
        def optimizer():
            return tf.train.GradientDescentOptimizer(0.1) 
            
        self.ps1 = ParameterServer(
            optimizer,
            {
                "x1": np.array([0.0, 0.0], dtype="float32"),
                "y1": np.array([1.0, 1.0], dtype="float32"),
            },
        )
        self.ps2 = ParameterServer(
            optimizer,
            {
                "x2": np.array([0.0, 0.0], dtype="float32"),
                "y2": np.array([1.0, 1.0], dtype="float32"),
            },
        )
        self.ps1.start()
        self.ps2.start()

    def tearDown(self):
        self.ps1.join()
        self.ps2.join()

    def test_all(self):
        # First non-blocking pull before any update.
        step, params = self.ps1.pull(blocking=False)
        self.assertEqual(step, 0)
        np.testing.assert_array_almost_equal(
            params["x1"], np.array([0.0, 0.0], dtype="float32")
        )
        np.testing.assert_array_almost_equal(
            params["y1"], np.array([1.0, 1.0], dtype="float32")
        )

        self.ps1.push(
            0,
            0,
            {
                "x1": np.array([0.1, 0.1], dtype="float32"),
                "y1": np.array([0.1, 0.1], dtype="float32"),
            },
        )

        # Pull the new version
        while step < 1:
            step, params = self.ps1.pull()
        self.assertEqual(step, 1)
        np.testing.assert_array_almost_equal(
            params["x1"], np.array([-0.01, -0.01], dtype="float32")
        )
        np.testing.assert_array_almost_equal(
            params["y1"], np.array([0.99, 0.99], dtype="float32")
        )

        # Pull by names
        step, params = self.ps1.pull(names=["x1"])
        self.assertEqual(step, 1)
        np.testing.assert_array_almost_equal(
            params["x1"], np.array([-0.01, -0.01], dtype="float32")
        )

        # Pull by unknown name
        with self.assertRaises(LookupError):
            self.ps1.pull(names=["z1"])
        # Pull a version non-blocking in future should raise exception
        with self.assertRaises(LookupError):
            self.ps1.pull(min_step=2, blocking=False)

        # Pull a version blocking in future should raise exception and blocks
        # for ~1s
        start = timer()
        with self.assertRaises(LookupError):
            self.ps1.pull(min_step=2, timeout=1.0)
        end = timer()
        self.assertGreater(end - start, 0.5)

        # Block indefinitely and push a new step from another thread with some
        # delay
        def waited_push():
            time.sleep(1)
            self.ps1.push(
                0,
                0,
                {
                    "x1": np.array([0.1, 0.1], dtype="float32"),
                    "y1": np.array([0.1, 0.1], dtype="float32"),
                },
            )

        t = threading.Thread(target=waited_push)
        t.start()

        start = timer()
        step, params = self.ps1.pull(min_step=2)
        end = timer()
        self.assertGreater(end - start, 0.5)

        self.assertEqual(step, 2)
        np.testing.assert_array_almost_equal(
            params["x1"], np.array([-0.02, -0.02], dtype="float32")
        )
        np.testing.assert_array_almost_equal(
            params["y1"], np.array([0.98, 0.98], dtype="float32")
        )
        t.join()


if __name__ == "__main__":
    unittest.main()
