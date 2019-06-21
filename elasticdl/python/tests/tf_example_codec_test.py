import unittest
import tensorflow as tf
import numpy as np
from elasticdl.python.data.codec import TFExampleCodec


class TestTFExampleCodec(unittest.TestCase):
    """ Test tf_example_codec.py
    """

    def _create_example(self, x, y):
        feature_name_to_feature = {}
        feature_name_to_feature['f0'] = tf.train.Feature(
            float_list=tf.train.FloatList(
                value=[x],
            ),
        )
        feature_name_to_feature['label'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[y]),
        )
        example = tf.train.Example(
            features=tf.train.Features(feature=feature_name_to_feature),
        )
        return example

    def test_encode_and_decode(self):
        expected = [(100.1, 1), (200.1, 2), (300.1, 3)]
        examples = [self._create_example(x[0], x[1]) for x in expected]

        # Create the codec for tf.train.Exampel data.
        codec = TFExampleCodec()

        # Encode
        encoded = [codec.encode(e) for e in examples]

        # feature_columns = [
        #     tf.feature_column.numeric_column(
        #         key="f0", dtype=tf.float32, shape=[1]
        #     ),
        #     tf.feature_column.numeric_column(
        #         key="label", dtype=tf.int64, shape=[1]
        #     ),
        # ]
        # example_spec = tf.feature_column.make_parse_example_spec(
        #     feature_columns
        # )

        # Verify decoded content.
        for idx, e in enumerate(encoded):
            example = codec.decode(e)
            # exp = codec.decode(e, example_spec)
            f_0 = example.features.feature["f0"].float_list.value[0]
            label = example.features.feature["label"].int64_list.value[0]
            self.assertEqual(f_0, np.float32(expected[idx][0]))
            self.assertEqual(label, np.int64(expected[idx][1]))


if __name__ == "__main__":
    unittest.main()
