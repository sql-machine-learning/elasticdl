import unittest
import tensorflow as tf
import numpy as np
from data.codec import BytesCodec


class TestBytesCodec(unittest.TestCase):
    """ Test bytes_codec.py
    """

    def test_encode_and_decode(self):
        feature_columns = [
            tf.feature_column.numeric_column(
                key="f0", dtype=tf.float64, shape=[1]
            ),
            tf.feature_column.numeric_column(
                key="label", dtype=tf.int64, shape=[1]
            ),
        ]

        example_1 = {"f0": np.array([100.1]), "label": np.array([1])}
        example_2 = {"f0": np.array([200.1]), "label": np.array([2])}
        example_3 = {"f0": np.array([300.1]), "label": np.array([3])}
        examples = [example_1, example_2, example_3]

        # Create the codec for tf.train.Exampel data.
        codec = BytesCodec(feature_columns)

        # Encode
        encoded = [codec.encode(e) for e in examples]

        # Verify decoded content
        for idx, e in enumerate(encoded):
            exp = codec.decode(e)
            expected_exp = examples[idx]
            self.assertEqual(exp["f0"], expected_exp["f0"])
            self.assertEqual(exp["label"], expected_exp["label"])


if __name__ == "__main__":
    unittest.main()
