import inspect
import os
import tempfile
import unittest

import numpy as np

from elasticdl.python.common.odps_recordio_conversion_utils import (
    _find_features_indices,
    _maybe_encode_unicode_string,
    _parse_row_to_example,
    write_recordio_shards_from_iterator,
)


class TestODPSRecordIOConversionUtils(unittest.TestCase):

    row1 = [61, 5.65, "Cash"]
    row2 = [50, 1.2, "Credit Card"]
    float_features = ["float_col"]
    bytes_features = ["bytes_col"]
    int_features = ["int_col"]
    features_list = ["int_col", "float_col", "bytes_col"]
    feature_indices = _find_features_indices(
        features_list, int_features, float_features, bytes_features
    )

    def test_parse_row_to_example(self):
        expected = inspect.cleandoc(
            """features {
      feature {
        key: "bytes_col"
        value {
          bytes_list {
            value: "Cash"
          }
        }
      }
      feature {
        key: "float_col"
        value {
          float_list {
            value: 5.650000095367432
          }
        }
      }
      feature {
        key: "int_col"
        value {
          int64_list {
            value: 61
          }
        }
      }
    }
    """
        )

        result = _parse_row_to_example(
            np.array(self.row1, dtype=object),
            self.features_list,
            self.feature_indices,
        )
        self.assertEqual(str(result), expected + "\n")

        result = _parse_row_to_example(
            self.row1, self.features_list, self.feature_indices
        )
        self.assertEqual(str(result), expected + "\n")

    def test_parse_row_to_example_unicode_mapping(self):
        self.assertEqual(
            _maybe_encode_unicode_string(u"some_unicode_string"),
            b"some_unicode_string",
        )
        self.assertEqual(
            _maybe_encode_unicode_string("some_string"), b"some_string"
        )

        _parse_row_to_example(
            np.array([u"40", u"2.5", "Chinese中文"], dtype=object),
            self.features_list,
            self.feature_indices,
        )

    def test_write_recordio_shards_from_iterator(self):
        features_list = ["Float1", "Float2", "Str1", "Int1"]
        # Each batch contains single item
        # records_per_shard = 1
        records_iter = iter(
            [[8.0, 10.65, "Cash", 6], [7.5, 17.8, "Credit Card", 3]]
        )
        with tempfile.TemporaryDirectory() as output_dir:
            write_recordio_shards_from_iterator(
                records_iter, features_list, output_dir, records_per_shard=1
            )
            self.assertEqual(
                os.listdir(output_dir), ["data-00000", "data-00001"]
            )

        # Each batch contains multiple items
        # records_per_shard = 1
        records_iter = iter(
            [
                [[1.0, 10.65, "Cash", 6], [2.5, 17.8, "Credit Card", 3]],
                [[3.0, 10.65, "Cash", 6], [4.5, 17.8, "Credit Card", 3]],
            ]
        )
        with tempfile.TemporaryDirectory() as output_dir:
            write_recordio_shards_from_iterator(
                records_iter, features_list, output_dir, records_per_shard=1
            )
            self.assertEqual(len(os.listdir(output_dir)), 4)

        # Each batch contains multiple items with fixed length
        # records_per_shard = 2
        records_iter = iter(
            [
                [[1.0, 10.65, "Cash", 6], [2.5, 17.8, "Credit Card", 3]],
                [[3.0, 10.65, "Cash", 6], [4.5, 17.8, "Credit Card", 3]],
            ]
        )
        with tempfile.TemporaryDirectory() as output_dir:
            write_recordio_shards_from_iterator(
                records_iter, features_list, output_dir, records_per_shard=2
            )
            self.assertEqual(len(os.listdir(output_dir)), 2)

        # Each batch contains multiple items with variable length
        # records_per_shard = 2
        records_iter = iter(
            [
                [[1.0, 10.65, "Cash", 6], [2.5, 17.8, "Credit Card", 3]],
                [[3.0, 10.65, "Cash", 6], [4.5, 17.8, "Credit Card", 3]],
                [[3.0, 10.65, "Cash", 6]],
            ]
        )
        with tempfile.TemporaryDirectory() as output_dir:
            write_recordio_shards_from_iterator(
                records_iter, features_list, output_dir, records_per_shard=2
            )
            self.assertEqual(len(os.listdir(output_dir)), 3)


if __name__ == "__main__":
    unittest.main()
