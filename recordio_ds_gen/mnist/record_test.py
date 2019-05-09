import unittest
from elasticdl.recordio_ds_gen.mnist import record
import numpy as np


class RecordTest(unittest.TestCase):
    def test_round_trip(self):
        # a random array
        d = np.ndarray(shape=(28, 28), dtype="uint8")
        encoded = record.encode(d, 5)
        dd, label = record.decode(encoded)
        np.testing.assert_array_equal(d, dd)
        self.assertEqual(label, 5)


if __name__ == "__main__":
    unittest.main()
