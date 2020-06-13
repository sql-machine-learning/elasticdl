# Copyright 2020 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
import tempfile
import unittest

import numpy as np

from elasticdl.python.data.odps_recordio_conversion_utils import (
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
        records_iter = iter(
            [[8.0, 10.65, "Cash", 6], [7.5, 17.8, "Credit Card", 3]]
        )
        with tempfile.TemporaryDirectory() as output_dir:
            write_recordio_shards_from_iterator(
                records_iter, features_list, output_dir, records_per_shard=1
            )
            self.assertEqual(
                sorted(os.listdir(output_dir)), ["data-00000", "data-00001"]
            )

        # Each batch contains multiple items
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
