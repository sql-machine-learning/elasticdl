import unittest
import tempfile
import os
import tensorflow as tf
import numpy as np
from elasticdl.data.coded_recordio import File
from elasticdl.data.tf_example_codec import TFExampleCodec


class TestTFExampleCodec(unittest.TestCase):
    """ Test tf_example_codec.py
    """

    def test_encode_and_decode(self):
        feature_schema = [
            ("f0", tf.string),
            ("f1", tf.float32),
            ("label", tf.int64),
        ]
        example_1 = [("f0", b"abc"), ("f1", 100.1), ("label", 1)]
        example_2 = [("f0", b"def"), ("f1", 200.1), ("label", 2)]
        example_3 = [("f0", b"ghi"), ("f1", 300.1), ("label", 3)]
        examples = [example_1, example_2, example_3]
        tmp_file = tempfile.NamedTemporaryFile(delete=False)

        # Create the codec for tf.train.Exampel data.
        codec = TFExampleCodec(feature_schema)

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
                    f_0, f_1, label = session.run(
                        [exp["f0"], exp["f1"], exp["label"]]
                    )
                    self.assertEqual(f_0, expected_exp[0][1])
                    self.assertEqual(f_1, np.float32(expected_exp[1][1]))
                    self.assertEqual(label, np.int64(expected_exp[2][1]))

        os.remove(tmp_file.name)


if __name__ == "__main__":
    unittest.main()
