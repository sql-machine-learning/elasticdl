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

import os
import random
import tempfile
import time
import unittest

from odps import ODPS

from elasticdl.python.common.constants import MaxComputeConfig
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
        self._project = os.environ[MaxComputeConfig.PROJECT_NAME]
        self._access_id = os.environ[MaxComputeConfig.ACCESS_ID]
        self._access_key = os.environ[MaxComputeConfig.ACCESS_KEY]
        self._endpoint = os.environ.get(MaxComputeConfig.ENDPOINT)
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

    def test_parallel_read(self):
        def transform(record):
            return float(record[0]) + 1

        start = 0
        end = 100
        shard_size = (end - start) // 4

        pd = ODPSReader(
            access_id=self._access_id,
            access_key=self._access_key,
            project=self._project,
            endpoint=self._endpoint,
            table=self._test_read_table,
            num_processes=2,
            transform_fn=transform,
        )

        results = []
        pd.reset((start, end - start), shard_size)
        shard_count = pd.get_shards_count()
        for i in range(shard_count):
            records = pd.get_records()
            for record in records:
                results.append(record)
        pd.stop()

        self.assertEqual(len(results), 100)

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
