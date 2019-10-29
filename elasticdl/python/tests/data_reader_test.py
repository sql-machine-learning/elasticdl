import os
import random
import tempfile
import time
import unittest
from collections import namedtuple

import numpy as np
import tensorflow as tf
from odps import ODPS

from elasticdl.python.common.constants import ODPSConfig
from elasticdl.python.common.model_utils import load_module
from elasticdl.python.data.data_reader import (
    Metadata,
    ODPSDataReader,
    RecordIODataReader,
    create_data_reader,
)
from elasticdl.python.tests.test_utils import (
    IRIS_TABLE_COLUMN_NAMES,
    DatasetName,
    create_iris_odps_table,
    create_recordio_file,
)

_MockedTask = namedtuple("Task", ["start", "end", "shard_name"])


class RecordIODataReaderTest(unittest.TestCase):
    def test_recordio_data_reader(self):
        num_records = 128
        with tempfile.TemporaryDirectory() as temp_dir_name:
            shard_name = create_recordio_file(
                num_records, DatasetName.TEST_MODULE, 1, temp_dir=temp_dir_name
            )

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
            self.test_table + ":shard_0": (0, self.records_per_task),
            self.test_table + ":shard_1": (50, self.records_per_task),
            self.test_table + ":shard_2": (100, 10),
        }
        self.assertEqual(expected_shards, self.reader.create_shards())

    def test_odps_data_reader_records_reading(self):
        records = list(
            self.reader.read_records(
                _MockedTask(0, 2, self.test_table + ":shard_0")
            )
        )
        records = np.array(records, dtype="float").tolist()
        self.assertEqual(
            [[6.4, 2.8, 5.6, 2.2, 2], [5.0, 2.3, 3.3, 1.0, 1]], records
        )
        self.assertEqual(
            self.reader.metadata.column_names, IRIS_TABLE_COLUMN_NAMES
        )

    def test_create_data_reader(self):
        reader = create_data_reader(
            data_origin="table", records_per_task=10, **{"columns": ["a", "b"]}
        )
        self.assertEqual(reader._kwargs["columns"], ["a", "b"])
        self.assertEqual(reader._kwargs["records_per_task"], 10)
        reader = create_data_reader(data_origin="table", records_per_task=10)
        self.assertEqual(reader._kwargs["records_per_task"], 10)
        self.assertTrue("columns" not in reader._kwargs)

    def test_odps_data_reader_integration_with_local_keras(self):
        num_records = 2
        model_spec = load_module(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "../../../model_zoo",
                "odps_iris_dnn_model/odps_iris_dnn_model.py",
            )
        ).__dict__
        model = model_spec["custom_model"]()
        optimizer = model_spec["optimizer"]()
        loss = model_spec["loss"]
        dataset_fn = model_spec["dataset_fn"]

        def _gen():
            for data in self.reader.read_records(
                _MockedTask(0, num_records, self.test_table + ":shard_0")
            ):
                if data is not None:
                    yield data

        dataset = tf.data.Dataset.from_generator(_gen, tf.string)
        dataset = dataset_fn(
            dataset, None, Metadata(column_names=IRIS_TABLE_COLUMN_NAMES)
        )
        dataset = dataset.batch(1)

        loss_history = []
        grads = None
        for features, labels in dataset:
            with tf.GradientTape() as tape:
                logits = model(features, training=True)
                loss_value = loss(logits, labels)
            loss_history.append(loss_value.numpy())
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        self.assertEqual(len(loss_history), num_records)
        self.assertEqual(len(grads), num_records)
        self.assertEqual(len(model.trainable_variables), num_records)

    def tearDown(self):
        self.odps_client.delete_table(
            self.test_table, self.project, if_exists=True
        )
