import os
import tempfile
import unittest

from elasticdl_preprocessing.utils import analyzer_utils


class AnalyzerUtilTest(unittest.TestCase):
    def test_methods_from_environment(self):
        # using default value
        self.assertEqual(analyzer_utils.get_min("age", 10), 10)
        self.assertEqual(analyzer_utils.get_max("age", 100), 100)
        self.assertEqual(analyzer_utils.get_avg("age", 10), 10)
        self.assertEqual(analyzer_utils.get_stddev("age", 10), 10)
        self.assertListEqual(
            analyzer_utils.get_bucket_boundaries("age", [10, 78]), [10, 78]
        )
        self.assertEqual(analyzer_utils.get_count("city", 19), 19)
        self.assertListEqual(
            analyzer_utils.get_vocabulary("city", ["a", "b"]), ["a", "b"]
        )

        # Get value from environment
        os.environ["age_min"] = "11"
        os.environ["age_max"] = "100"
        os.environ["age_avg"] = "50.9"
        os.environ["age_stddev"] = "90.87"
        os.environ["age_bkt_boundaries"] = "15,67,89"
        os.environ["city_count"] = "50"

        self.assertEqual(analyzer_utils.get_min("age", 10), 11)
        self.assertEqual(analyzer_utils.get_max("age", 10), 100)
        self.assertEqual(analyzer_utils.get_avg("age", 10), 50.9)
        self.assertEqual(analyzer_utils.get_stddev("age", 10), 90.87)
        self.assertListEqual(
            analyzer_utils.get_bucket_boundaries("age", [10, 78]), [15, 67, 89]
        )
        self.assertEqual(analyzer_utils.get_count("city", 19), 50)

        with tempfile.TemporaryDirectory() as temp_dir:
            vocab_path = os.path.join(temp_dir, "city.txt")

            with open(vocab_path, "w") as f:
                f.write("Beijing\n")
                f.write("NewYork\n")
            os.environ["city_vocab_path"] = vocab_path
            self.assertListEqual(
                analyzer_utils.get_vocabulary("city", ["a", "b"]),
                ["Beijing", "NewYork"],
            )
