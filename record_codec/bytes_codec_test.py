import unittest
from elasticdl.record_codec.bytes_codec import BytesCodec 
import numpy as np


class RecordTest(unittest.TestCase):
    def test_round_trip(self):
        # a random array
        codec = BytesCodec()
        d = np.ndarray(shape=(28, 28), dtype="uint8")
        data = [('image', d), ('label', 5)]
        encoded = codec.encode(data)
        decoded = codec.decode(encoded)
        np.testing.assert_array_equal(d, decoded['image'])
        self.assertEqual(label, decoded['label'])


if __name__ == "__main__":
    unittest.main()
