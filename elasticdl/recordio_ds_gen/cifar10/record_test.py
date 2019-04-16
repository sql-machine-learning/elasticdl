import unittest
import record
import numpy as np

N = 32

class RecordTest(unittest.TestCase):
    def test_round_trip(self):
        img = np.ndarray(shape=(3, N, N), dtype="uint8")
        red = img[0]
        green = img[1]
        blue = img[2]
        # build an image with a recognizable pattern.
        for i in range(N):
            for j in range(N):
                red[i, j] = 8 * i
                green[i,j] = (i + j) * 4
                blue[i, j] = 8 * j
        encoded = record.encode(img, 5)
        d_img, d_label = record.decode(encoded)
        np.testing.assert_array_equal(img, d_img)
        self.assertEqual(d_label, 5)

        record.show(d_img, d_label)


if __name__ == '__main__':
    unittest.main()
