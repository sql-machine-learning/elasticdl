import unittest
import tensorflow as tf
import numpy as np
from elasticdl.python.data.codec import TFExampleCodec


class TestTFExampleCodec(unittest.TestCase):
    """ Test tf_example_codec.py
    """

    def test_encode_and_decode(self):
        feature_columns = [
            tf.feature_column.numeric_column(
                key="f0", dtype=tf.float32, shape=[1]
            ),
            tf.feature_column.numeric_column(
                key="label", dtype=tf.int64, shape=[1]
            ),
        ]

        feature_name_to_type = {
            f_col.key: f_col.dtype for f_col in feature_columns
        }

        example_spec = tf.feature_column.make_parse_example_spec(
            feature_columns,
        )

        example_1 = {"f0": np.array(100.1), "label": np.array(1)}
        example_2 = {"f0": np.array(200.1), "label": np.array(2)}
        example_3 = {"f0": np.array(300.1), "label": np.array(3)}
        examples = [example_1, example_2, example_3]

        # Create the codec for tf.train.Exampel data.
        codec = TFExampleCodec()

        # Encode
        encoded = [codec.encode(e, feature_name_to_type) for e in examples]

        # Verify decoded content.
        for idx, e in enumerate(encoded):
            exp = codec.decode(e, example_spec)
            expected_exp = examples[idx]
            f_0 = exp["f0"].numpy()
            label = exp["label"].numpy()
            self.assertEqual(f_0, np.float32(expected_exp["f0"]))
            self.assertEqual(label, np.int64(expected_exp["label"]))


if __name__ == "__main__":
    unittest.main()
