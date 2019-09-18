import os
import random
import tempfile
import time
import unittest
from collections import namedtuple
from contextlib import closing

import numpy as np
import recordio
import tensorflow as tf
from odps import ODPS

from elasticdl.python.common.constants import ODPSConfig
from elasticdl.python.common.data_reader import (
    ODPSDataReader,
    RecordIODataReader,
)
from elasticdl.python.tests.odps_test_utils import create_iris_odps_table


def _create_recordio_file(size, temp_dir):
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
    with closing(recordio.Writer(temp_file.name)) as f:
        for _ in range(size):
            x = np.random.rand(1).astype(np.float32)
            y = 2 * x + 1
            example_dict = {
                "x": tf.train.Feature(float_list=tf.train.FloatList(value=x)),
                "y": tf.train.Feature(float_list=tf.train.FloatList(value=y)),
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=example_dict)
            )
            f.write(example.SerializeToString())
    return temp_file.name


class DataReaderTest(unittest.TestCase):
    Task = namedtuple("Task", ["start", "end", "shard_name"])

    def test_recordio_data_reader(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            shard_name = _create_recordio_file(128, temp_dir_name)

            # Test shards creation
            expected_shards = {shard_name: (0, 128)}
            reader = RecordIODataReader(data_dir=temp_dir_name)
            self.assertEqual(expected_shards, reader.create_shards())

            # Test records reading
            records = list(reader.read_records(self.Task(0, 2, shard_name)))
            self.assertEqual(len(records), 2)
            for record in records:
                parsed_record = tf.io.parse_single_example(
                    record,
                    {
                        "x": tf.io.FixedLenFeature([1], tf.float32),
                        "y": tf.io.FixedLenFeature([1], tf.float32),
                    },
                )
                for k, v in parsed_record.items():
                    self.assertEqual(len(v.numpy()), 1)

    @unittest.skipIf(
        os.environ.get("ODPS_TESTS", "False") == "False",
        "ODPS environment is not configured",
    )
    def test_odps_data_reader(self):
        # Setup
        project = os.environ[ODPSConfig.PROJECT_NAME]
        access_id = os.environ[ODPSConfig.ACCESS_ID]
        access_key = os.environ[ODPSConfig.ACCESS_KEY]
        endpoint = os.environ.get(ODPSConfig.ENDPOINT)
        test_table = "test_odps_data_reader_%d_%d" % (
            int(time.time()),
            random.randint(1, 101),
        )
        odps_client = ODPS(access_id, access_key, project, endpoint)
        create_iris_odps_table(odps_client, project, test_table)

        reader = ODPSDataReader(
            project=project,
            access_id=access_id,
            access_key=access_key,
            endpoint=endpoint,
            table=test_table,
            num_processes=1,
            records_per_task=50,
        )

        # Test shards creation
        expected_shards = {
            "shard_0": (0, 49),
            "shard_1": (50, 99),
            "shard_2": (100, 110),
        }
        self.assertEqual(expected_shards, reader.create_shards())

        # Test records reading
        records = list(reader.read_records(self.Task(0, 2, "shard_0")))
        self.assertEqual(
            [[6.4, 2.8, 5.6, 2.2, 2], [5.0, 2.3, 3.3, 1.0, 1]], records
        )

        # Teardown
        odps_client.delete_table(test_table, project, if_exists=True)
