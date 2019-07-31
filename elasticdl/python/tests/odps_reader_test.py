import os
import unittest
import tempfile

from elasticdl.python.common.constants import ODPSConfig
from elasticdl.python.common.odps_reader import ODPSReader
from elasticdl.python.common.odps_recordio_conversion_utils import (
    write_recordio_shards_from_iterator,
)


@unittest.skipIf(
    os.environ.get("ODPS_TESTS", "False") == "False",
    "ODPS environment is not configured",
)
class ODPSReaderTest(unittest.TestCase):
    def setUp(self):
        self._project = os.environ[ODPSConfig.PROJECT_NAME]
        self._access_id = os.environ[ODPSConfig.ACCESS_ID]
        self._access_key = os.environ[ODPSConfig.ACCESS_KEY]
        self._endpoint = os.environ[ODPSConfig.ENDPOINT]
        self._table = "chicago_taxi_train_data"
        self.reader = ODPSReader(
            self._project,
            self._access_id,
            self._access_key,
            self._endpoint,
            self._table,
            None,
            4,
            None,
        )

    def test_to_iterator(self):
        records_iter = self.reader.to_iterator(1, 0, 200, 2, False, None)
        for batch in records_iter:
            self.assertEqual(
                len(batch), 200, "incompatible size: %d" % len(batch)
            )

    def test_write_odpsrecordio_shards_from_iterator(self):
        records_iter = self.reader.to_iterator(1, 0, 200, 2, False, None)
        with tempfile.TemporaryDirectory() as output_dir:
            write_recordio_shards_from_iterator(
                records_iter,
                ['f' + str(i) for i in range(18)],
                output_dir,
                records_per_shard=100,
            )
            self.assertEqual(
                len(os.listdir(output_dir)), 100
            )


if __name__ == "__main__":
    unittest.main()
