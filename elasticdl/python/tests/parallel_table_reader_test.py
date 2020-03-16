import os
import random
import time
import unittest

from odps import ODPS

from elasticdl.python.common.constants import MaxComputeConfig
from elasticdl.python.data.odps_io import is_odps_configured
from elasticdl.python.data.parallel_table_reader import ParallelTableReader
from elasticdl.python.tests.test_utils import create_iris_odps_table


@unittest.skipIf(
    not is_odps_configured(), "ODPS environment is not configured",
)
class ParallelTableReaderTest(unittest.TestCase):
    def setUp(self):
        self._project = os.environ[MaxComputeConfig.PROJECT_NAME]
        self._access_id = os.environ[MaxComputeConfig.ACCESS_ID]
        self._access_key = os.environ[MaxComputeConfig.ACCESS_KEY]
        self._endpoint = os.environ.get(MaxComputeConfig.ENDPOINT)
        self._test_read_table = "test_odps_reader_%d_%d" % (
            int(time.time()),
            random.randint(1, 101),
        )
        self._odps_client = ODPS(
            self._access_id, self._access_key, self._project, self._endpoint
        )
        create_iris_odps_table(
            self._odps_client, self._project, self._test_read_table
        )

    def tearDown(self):
        self._odps_client.delete_table(
            self._test_read_table, self._project, if_exists=True
        )

    def test_parallel_read(self):
        def transform(record):
            return float(record[0]) + 1

        start = 0
        end = 100
        batch_size = (end - start) // 4
        num_parallel_processes = 2

        pd = ParallelTableReader(
            self._access_id,
            self._access_key,
            self._project,
            self._endpoint,
            self._test_read_table,
            "",
            None,
            batch_size,
            num_parallel_processes,
            transform,
        )

        results = []
        pd.reset((start, end - start))
        shard_count = pd.get_shards_count()
        for i in range(shard_count):
            records = pd.get_records()
            for record in records:
                results.append(record)
        pd.stop()

        self.assertEqual(len(results), 100)


if __name__ == "__main__":
    unittest.main()
