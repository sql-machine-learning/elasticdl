import unittest
import tempfile
import os
import tensorflow as tf
import numpy as np
from tf_example_file import TFExampleFile 

class TestTFExampleFile(unittest.TestCase):
    """ Test tf_example_file.py
    """

    def test_write_and_get(self):
        feature_schemas = [("feature0", tf.string), ("feature1", tf.float32), ("label", tf.int64)] 
        tmp_file = tempfile.NamedTemporaryFile(delete=False)

        with TFExampleFile(tmp_file.name, feature_schemas, 'w') as rdio_w:
            features = [("feature0", b"abc"), ("feature1", 100.1), ("label", 1)] 
            rdio_w.write(features)
            
        with TFExampleFile(tmp_file.name, feature_schemas, 'r') as rdio_r:
            features_1 = rdio_r.get(0)
            with tf.Session() as session:
                f_0, f_1, f_2 = session.run([features_1["feature0"], features_1["feature1"], features_1["label"]])
                self.assertEqual(f_0, b"abc")
                self.assertEqual(f_1, np.float32(100.1))
                self.assertEqual(f_2, np.int64(1))

        os.remove(tmp_file.name)

    def test_iterate(self):
        feature_schemas = [("feature0", tf.string), ("feature1", tf.float32), ("label", tf.int64)]
        tmp_file = tempfile.NamedTemporaryFile(delete=False)

        with TFExampleFile(tmp_file.name, feature_schemas, 'w') as rdio_w:
            features_1 = [("feature0", b"abc"), ("feature1", 100.1), ("label", 1)]
            features_2 = [("feature0", b"def"), ("feature1", 200.1), ("label", 2)]
            features_3 = [("feature0", b"ghi"), ("feature1", 300.1), ("label", 3)]
            features_all = [features_1, features_2, features_3]
            rdio_w.write(features_1)
            rdio_w.write(features_2)
            rdio_w.write(features_3)

        with TFExampleFile(tmp_file.name, feature_schemas, 'r') as rdio_r:
            with tf.Session() as session:
                for idx, features in enumerate(rdio_r):
                    exp_f = features_all[idx]
                    f_0, f_1, f_2 = session.run([features["feature0"], features["feature1"], features["label"]])
                    self.assertEqual(f_0, exp_f[0][1])
                    self.assertEqual(f_1, np.float32(exp_f[1][1]))
                    self.assertEqual(f_2, np.int64(exp_f[2][1]))

        os.remove(tmp_file.name)

if __name__ == '__main__':
    unittest.main()
