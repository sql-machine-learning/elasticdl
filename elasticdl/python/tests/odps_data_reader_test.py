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
from elasticdl.python.common.model_helper import load_module
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


_MockedTask = namedtuple("Task", ["start", "end", "shard_name"])


class RecordIODataReaderTest(unittest.TestCase):
    def test_recordio_data_reader(self):
        num_records = 128
        with tempfile.TemporaryDirectory() as temp_dir_name:
            shard_name = _create_recordio_file(num_records, temp_dir_name)

            # Test shards creation
            expected_shards = {shard_name: (0, num_records)}
            reader = RecordIODataReader(data_dir=temp_dir_name)
            self.assertEqual(expected_shards, reader.create_shards())

            # Test records reading
            records = list(
                reader.read_records(_MockedTask(0, num_records, shard_name))
            )
            self.assertEqual(len(records), num_records)
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
class ODPSDataReaderTest(unittest.TestCase):
    def setUp(self):
        self.project = os.environ[ODPSConfig.PROJECT_NAME]
        access_id = os.environ[ODPSConfig.ACCESS_ID]
        access_key = os.environ[ODPSConfig.ACCESS_KEY]
        endpoint = os.environ.get(ODPSConfig.ENDPOINT)
        self.test_table = "test_odps_data_reader_%d_%d" % (
            int(time.time()),
            random.randint(1, 101),
        )
        self.odps_client = ODPS(access_id, access_key, self.project, endpoint)
        create_iris_odps_table(self.odps_client, self.project, self.test_table)
        self.records_per_task = 50

        self.reader = ODPSDataReader(
            project=self.project,
            access_id=access_id,
            access_key=access_key,
            endpoint=endpoint,
            table=self.test_table,
            num_processes=1,
            records_per_task=self.records_per_task,
        )

    def test_odps_data_reader_shards_creation(self):
        expected_shards = {
            "shard_0": (0, self.records_per_task),
            "shard_1": (50, self.records_per_task),
            "shard_2": (100, 10),
        }
        self.assertEqual(expected_shards, self.reader.create_shards())

    def test_odps_data_reader_records_reading(self):
        records = list(self.reader.read_records(_MockedTask(0, 2, "shard_0")))
        self.assertEqual(
            [[6.4, 2.8, 5.6, 2.2, 2], [5.0, 2.3, 3.3, 1.0, 1]], records
        )

    def test_odps_data_reader_integration_with_local_keras(self):
        model_spec = load_module(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "odps_test_module.py",
            )
        ).__dict__
        model = model_spec["custom_model"]()
        optimizer = model_spec["optimizer"]()
        loss = model_spec["loss"]
        dataset_fn = model_spec["dataset_fn"]

        def _gen():
            for data in self.reader.read_records(_MockedTask(0, 2, "shard_0")):
                if data is not None:
                    yield data

        dataset = tf.data.Dataset.from_generator(_gen, (tf.float32))
        dataset = dataset_fn(dataset, None)

        loss_history = []
        for features, labels in dataset:
            with tf.GradientTape() as tape:
                logits = model(features, training=True)
                loss_value = loss(logits, labels)
            loss_history.append(loss_value.numpy())
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    def tearDown(self):
        self.odps_client.delete_table(
            self.test_table, self.project, if_exists=True
        )
