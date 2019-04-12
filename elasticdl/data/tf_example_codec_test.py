import unittest
import tempfile
import os
import tensorflow as tf
import numpy as np
from .coded_recordio import File
from .tf_example_codec import TFExampleEncoder, TFExampleDecoder 

class TestTFExampleCodec(unittest.TestCase):
    """ Test tf_example_codec.py
    """

    def test_encode_and_decode(self):
        feature_schema = [("feature0", tf.string), ("feature1", tf.float32), ("label", tf.int64)]
        example_1 = [("feature0", b"abc"), ("feature1", 100.1), ("label", 1)]
        example_2 = [("feature0", b"def"), ("feature1", 200.1), ("label", 2)]
        example_3 = [("feature0", b"ghi"), ("feature1", 300.1), ("label", 3)]
        examples = [example_1, example_2, example_3]
        tmp_file = tempfile.NamedTemporaryFile(delete=False)

        # Create encoder and decoder.
        encoder = TFExampleEncoder(feature_schema)
        decoder = TFExampleDecoder(feature_schema)

        # Write an encoded RecordIO file.
        with File(tmp_file.name, "w", encoder=encoder) as coded_w:
            for example in examples:
                coded_w.write(example)

        # Verify decoded content, with get() interface.
        with File(tmp_file.name, "r", decoder=decoder) as coded_r:
            with tf.Session() as session:
                for idx in range(coded_r.count()):
                    example = coded_r.get(idx)
                    expected_exp = examples[idx]
                    f_0, f_1, f_2 = session.run([example["feature0"], example["feature1"], example["label"]])
                    self.assertEqual(f_0, expected_exp[0][1])
                    self.assertEqual(f_1, np.float32(expected_exp[1][1]))
                    self.assertEqual(f_2, np.int64(expected_exp[2][1]))

        os.remove(tmp_file.name)

if __name__ == '__main__':
    unittest.main()
