import os
import random
import tempfile
import time
import unittest

from odps import ODPS

from elasticdl.python.common.constants import ODPSConfig
from elasticdl.python.data.odps_io import (
    ODPSReader,
    ODPSWriter,
    is_odps_configured,
)
from elasticdl.python.data.odps_recordio_conversion_utils import (
    write_recordio_shards_from_iterator,
)
from elasticdl.python.tests.test_utils import create_iris_odps_table


@unittest.skipIf(
    not is_odps_configured(), "ODPS environment is not configured",
)
class ODPSIOTest(unittest.TestCase):
    def setUp(self):
        self._project = os.environ[ODPSConfig.PROJECT_NAME]
        self._access_id = os.environ[ODPSConfig.ACCESS_ID]
        self._access_key = os.environ[ODPSConfig.ACCESS_KEY]
        self._endpoint = os.environ.get(ODPSConfig.ENDPOINT)
        self._test_read_table = "test_odps_reader_%d_%d" % (
            int(time.time()),
            random.randint(1, 101),
        )
        self._test_write_table = "test_odps_writer_%d_%d" % (
            int(time.time()),
            random.randint(1, 101),
        )
        self._odps_client = ODPS(
            self._access_id, self._access_key, self._project, self._endpoint
        )
        create_iris_odps_table(
            self._odps_client, self._project, self._test_read_table
        )

    def test_read_to_iterator(self):
        reader = ODPSReader(
            self._project,
            self._access_id,
            self._access_key,
            self._endpoint,
            self._test_read_table,
            None,
            4,
            None,
        )
        records_iter = reader.to_iterator(1, 0, 50, 2, False, None)
        records = list(records_iter)
        self.assertEqual(
            len(records), 6, "Unexpected number of batches: %d" % len(records)
        )
        flattened_records = [record for batch in records for record in batch]
        self.assertEqual(
            len(flattened_records),
            220,
            "Unexpected number of total records: %d" % len(flattened_records),
        )

    def test_write_odps_to_recordio_shards_from_iterator(self):
        reader = ODPSReader(
            self._project,
            self._access_id,
            self._access_key,
            self._endpoint,
            self._test_read_table,
            None,
            4,
            None,
        )
        records_iter = reader.to_iterator(1, 0, 50, 2, False, None)
        with tempfile.TemporaryDirectory() as output_dir:
            write_recordio_shards_from_iterator(
                records_iter,
                ["f" + str(i) for i in range(5)],
                output_dir,
                records_per_shard=50,
            )
            self.assertEqual(len(os.listdir(output_dir)), 5)

    def test_write_from_iterator(self):
        columns = ["num", "num2"]
        column_types = ["bigint", "double"]

        # If the table doesn't exist yet
        writer = ODPSWriter(
            self._project,
            self._access_id,
            self._access_key,
            self._endpoint,
            self._test_write_table,
            columns,
            column_types,
        )
        writer.from_iterator(iter([[1, 0.5], [2, 0.6]]), 2)
        table = self._odps_client.get_table(
            self._test_write_table, self._project
        )
        self.assertEqual(table.schema.names, columns)
        self.assertEqual(table.schema.types, column_types)
        self.assertEqual(table.to_df().count(), 1)

        # If the table already exists
        writer = ODPSWriter(
            self._project,
            self._access_id,
            self._access_key,
            self._endpoint,
            self._test_write_table,
        )
        writer.from_iterator(iter([[1, 0.5], [2, 0.6]]), 2)
        table = self._odps_client.get_table(
            self._test_write_table, self._project
        )
        self.assertEqual(table.schema.names, columns)
        self.assertEqual(table.schema.types, column_types)
        self.assertEqual(table.to_df().count(), 2)

    def tearDown(self):
        self._odps_client.delete_table(
            self._test_write_table, self._project, if_exists=True
        )
        self._odps_client.delete_table(
            self._test_read_table, self._project, if_exists=True
        )


if __name__ == "__main__":
    unittest.main()
