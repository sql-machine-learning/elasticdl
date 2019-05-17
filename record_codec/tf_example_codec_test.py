import unittest
import tempfile
import os
import tensorflow as tf
import numpy as np
from recordio import File
from record_codec import TFExampleCodec


class TestTFExampleCodec(unittest.TestCase):
    """ Test tf_example_codec.py
    """

    def test_encode_and_decode(self):
        feature_columns = [tf.feature_column.numeric_column(key="f0",
            dtype=tf.float32, shape=[1]),
            tf.feature_column.numeric_column(key="label",
            dtype=tf.int64, shape=[1])]

        example_1 = [("f0", np.array(100.1)), ("label", np.array(1))]
        example_2 = [("f0", np.array(200.1)), ("label", np.array(2))]
        example_3 = [("f0", np.array(300.1)), ("label", np.array(3))]
        examples = [example_1, example_2, example_3]

        tmp_file = tempfile.NamedTemporaryFile(delete=False)

        # Create the codec for tf.train.Exampel data.
        codec = TFExampleCodec(feature_columns)

        # Write an encoded RecordIO file.
        with File(tmp_file.name, "w", encoder=codec.encode) as coded_w:
            for example in examples:
                coded_w.write(example)

        # Verify decoded content, with get() interface.
        with File(tmp_file.name, "r", decoder=codec.decode) as coded_r:
            with tf.Session() as session:
                for idx in range(coded_r.count()):
                    exp = coded_r.get(idx)
                    expected_exp = examples[idx]
                    f_0, label = session.run(
                        [exp["f0"], exp["label"]]
                    )
                    self.assertEqual(f_0, np.float32(expected_exp[0][1]))
                    self.assertEqual(label, np.int64(expected_exp[1][1]))

        os.remove(tmp_file.name)


if __name__ == "__main__":
    unittest.main()
