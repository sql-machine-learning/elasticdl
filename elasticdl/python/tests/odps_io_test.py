import os
import random
import tempfile
import time
import unittest

from odps import ODPS

from elasticdl.python.common.constants import ODPSConfig
from elasticdl.python.common.odps_io import ODPSReader, ODPSWriter
from elasticdl.python.common.odps_recordio_conversion_utils import (
    write_recordio_shards_from_iterator,
)


@unittest.skipIf(
    os.environ.get("ODPS_TESTS", "False") == "False",
    "ODPS environment is not configured",
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
        self.create_iris_odps_table()

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
        records_iter = reader.to_iterator(1, 0, 200, 2, False, None)
        for batch in records_iter:
            self.assertEqual(
                len(batch), 200, "incompatible size: %d" % len(batch)
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
        records_iter = reader.to_iterator(1, 0, 200, 2, False, None)
        with tempfile.TemporaryDirectory() as output_dir:
            write_recordio_shards_from_iterator(
                records_iter,
                ["f" + str(i) for i in range(18)],
                output_dir,
                records_per_shard=200,
            )
            self.assertEqual(len(os.listdir(output_dir)), 100)

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

    def create_iris_odps_table(self):
        sql_tmpl = """
        DROP TABLE IF EXISTS {PROJECT_NAME}.{TABLE_NAME};
        CREATE TABLE {PROJECT_NAME}.{TABLE_NAME} (
               sepal_length DOUBLE,
               sepal_width  DOUBLE,
               petal_length DOUBLE,
               petal_width  DOUBLE,
               class BIGINT);

        INSERT INTO {PROJECT_NAME}.{TABLE_NAME} VALUES
        (6.4,2.8,5.6,2.2,2),
        (5.0,2.3,3.3,1.0,1),
        (4.9,2.5,4.5,1.7,2),
        (4.9,3.1,1.5,0.1,0),
        (5.7,3.8,1.7,0.3,0),
        (4.4,3.2,1.3,0.2,0),
        (5.4,3.4,1.5,0.4,0),
        (6.9,3.1,5.1,2.3,2),
        (6.7,3.1,4.4,1.4,1),
        (5.1,3.7,1.5,0.4,0),
        (5.2,2.7,3.9,1.4,1),
        (6.9,3.1,4.9,1.5,1),
        (5.8,4.0,1.2,0.2,0),
        (5.4,3.9,1.7,0.4,0),
        (7.7,3.8,6.7,2.2,2),
        (6.3,3.3,4.7,1.6,1),
        (6.8,3.2,5.9,2.3,2),
        (7.6,3.0,6.6,2.1,2),
        (6.4,3.2,5.3,2.3,2),
        (5.7,4.4,1.5,0.4,0),
        (6.7,3.3,5.7,2.1,2),
        (6.4,2.8,5.6,2.1,2),
        (5.4,3.9,1.3,0.4,0),
        (6.1,2.6,5.6,1.4,2),
        (7.2,3.0,5.8,1.6,2),
        (5.2,3.5,1.5,0.2,0),
        (5.8,2.6,4.0,1.2,1),
        (5.9,3.0,5.1,1.8,2),
        (5.4,3.0,4.5,1.5,1),
        (6.7,3.0,5.0,1.7,1),
        (6.3,2.3,4.4,1.3,1),
        (5.1,2.5,3.0,1.1,1),
        (6.4,3.2,4.5,1.5,1),
        (6.8,3.0,5.5,2.1,2),
        (6.2,2.8,4.8,1.8,2),
        (6.9,3.2,5.7,2.3,2),
        (6.5,3.2,5.1,2.0,2),
        (5.8,2.8,5.1,2.4,2),
        (5.1,3.8,1.5,0.3,0),
        (4.8,3.0,1.4,0.3,0),
        (7.9,3.8,6.4,2.0,2),
        (5.8,2.7,5.1,1.9,2),
        (6.7,3.0,5.2,2.3,2),
        (5.1,3.8,1.9,0.4,0),
        (4.7,3.2,1.6,0.2,0),
        (6.0,2.2,5.0,1.5,2),
        (4.8,3.4,1.6,0.2,0),
        (7.7,2.6,6.9,2.3,2),
        (4.6,3.6,1.0,0.2,0),
        (7.2,3.2,6.0,1.8,2),
        (5.0,3.3,1.4,0.2,0),
        (6.6,3.0,4.4,1.4,1),
        (6.1,2.8,4.0,1.3,1),
        (5.0,3.2,1.2,0.2,0),
        (7.0,3.2,4.7,1.4,1),
        (6.0,3.0,4.8,1.8,2),
        (7.4,2.8,6.1,1.9,2),
        (5.8,2.7,5.1,1.9,2),
        (6.2,3.4,5.4,2.3,2),
        (5.0,2.0,3.5,1.0,1),
        (5.6,2.5,3.9,1.1,1),
        (6.7,3.1,5.6,2.4,2),
        (6.3,2.5,5.0,1.9,2),
        (6.4,3.1,5.5,1.8,2),
        (6.2,2.2,4.5,1.5,1),
        (7.3,2.9,6.3,1.8,2),
        (4.4,3.0,1.3,0.2,0),
        (7.2,3.6,6.1,2.5,2),
        (6.5,3.0,5.5,1.8,2),
        (5.0,3.4,1.5,0.2,0),
        (4.7,3.2,1.3,0.2,0),
        (6.6,2.9,4.6,1.3,1),
        (5.5,3.5,1.3,0.2,0),
        (7.7,3.0,6.1,2.3,2),
        (6.1,3.0,4.9,1.8,2),
        (4.9,3.1,1.5,0.1,0),
        (5.5,2.4,3.8,1.1,1),
        (5.7,2.9,4.2,1.3,1),
        (6.0,2.9,4.5,1.5,1),
        (6.4,2.7,5.3,1.9,2),
        (5.4,3.7,1.5,0.2,0),
        (6.1,2.9,4.7,1.4,1),
        (6.5,2.8,4.6,1.5,1),
        (5.6,2.7,4.2,1.3,1),
        (6.3,3.4,5.6,2.4,2),
        (4.9,3.1,1.5,0.1,0),
        (6.8,2.8,4.8,1.4,1),
        (5.7,2.8,4.5,1.3,1),
        (6.0,2.7,5.1,1.6,1),
        (5.0,3.5,1.3,0.3,0),
        (6.5,3.0,5.2,2.0,2),
        (6.1,2.8,4.7,1.2,1),
        (5.1,3.5,1.4,0.3,0),
        (4.6,3.1,1.5,0.2,0),
        (6.5,3.0,5.8,2.2,2),
        (4.6,3.4,1.4,0.3,0),
        (4.6,3.2,1.4,0.2,0),
        (7.7,2.8,6.7,2.0,2),
        (5.9,3.2,4.8,1.8,1),
        (5.1,3.8,1.6,0.2,0),
        (4.9,3.0,1.4,0.2,0),
        (4.9,2.4,3.3,1.0,1),
        (4.5,2.3,1.3,0.3,0),
        (5.8,2.7,4.1,1.0,1),
        (5.0,3.4,1.6,0.4,0),
        (5.2,3.4,1.4,0.2,0),
        (5.3,3.7,1.5,0.2,0),
        (5.0,3.6,1.4,0.2,0),
        (5.6,2.9,3.6,1.3,1),
        (4.8,3.1,1.6,0.2,0);
        """
        self._odps_client.execute_sql(
            sql_tmpl.format(
                PROJECT_NAME=self._project,
                TABLE_NAME=self._test_read_table,
            ),
            hints={"odps.sql.submit.mode": "script"}
        )


if __name__ == "__main__":
    unittest.main()
