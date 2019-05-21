import unittest
import tempfile
import os
import tensorflow as tf
import numpy as np
from recordio import File
from edl_data.codec  import BytesCodec


class TestBytesCodec(unittest.TestCase):
    """ Test bytes_codec.py
    """

    def test_encode_and_decode(self):
        feature_columns = [tf.feature_column.numeric_column(key="f0",
            dtype=tf.float64, shape=[1]),
            tf.feature_column.numeric_column(key="label",
            dtype=tf.int64, shape=[1])]

        example_1 = {"f0": np.array([100.1]), "label": np.array([1])}
        example_2 = {"f0": np.array([200.1]), "label": np.array([2])}
        example_3 = {"f0": np.array([300.1]), "label": np.array([3])}
        examples = [example_1, example_2, example_3]

        tmp_file = tempfile.NamedTemporaryFile(delete=False)

        # Create the codec for tf.train.Exampel data.
        codec = BytesCodec(feature_columns)

        # Write an encoded RecordIO file.
        with File(tmp_file.name, "w", encoder=codec.encode) as coded_w:
            for example in examples:
                coded_w.write(example)

        # Verify decoded content, with get() interface.
        with File(tmp_file.name, "r", decoder=codec.decode) as coded_r:
            for idx in range(coded_r.count()):
                exp = coded_r.get(idx)
                expected_exp = examples[idx]
                self.assertEqual(exp["f0"], expected_exp["f0"])
                self.assertEqual(exp["label"], expected_exp["label"])

        os.remove(tmp_file.name)


if __name__ == "__main__":
    unittest.main()
