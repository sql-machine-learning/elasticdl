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

import numpy as np
import odps
import tensorflow as tf
from odps import ODPS

from elasticai_api.proto import elasticai_api_pb2
from elasticdl.python.common.constants import MaxComputeConfig
from elasticdl.python.common.model_utils import load_module
from elasticdl.python.data.odps_io import is_odps_configured
from elasticdl.python.data.reader.data_reader import Metadata
from elasticdl.python.data.reader.data_reader_factory import create_data_reader
from elasticdl.python.data.reader.odps_reader import ODPSDataReader
from elasticdl.python.data.reader.recordio_reader import RecordIODataReader
from elasticdl.python.data.reader.text_reader import TextDataReader
from elasticdl.python.master.task_manager import _Task
from elasticdl.python.tests.test_utils import (
    IRIS_TABLE_COLUMN_NAMES,
    DatasetName,
    create_iris_csv_file,
    create_iris_odps_table,
    create_recordio_file,
)


class RecordIODataReaderTest(unittest.TestCase):
    def test_recordio_data_reader(self):
        num_records = 128
        with tempfile.TemporaryDirectory() as temp_dir_name:
            shard_name = create_recordio_file(
                num_records, DatasetName.TEST_MODULE, 1, temp_dir=temp_dir_name
            )

            # Test shards creation
            expected_shards = [(shard_name, 0, num_records)]
            reader = RecordIODataReader(data_dir=temp_dir_name)
            self.assertEqual(expected_shards, reader.create_shards())

            # Test records reading
            records = list(
                reader.read_records(
                    _Task(
                        shard_name, 0, num_records, elasticai_api_pb2.TRAINING
                    )
                )
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


class TextDataReaderTest(unittest.TestCase):
    def test_csv_data_reader(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            num_records = 128
            columns = [
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
                "class",
            ]
            iris_file_name = create_iris_csv_file(
                size=num_records, columns=columns, temp_dir=temp_dir_name
            )
            csv_data_reader = TextDataReader(
                filename=iris_file_name, records_per_task=20
            )
            shards = csv_data_reader.create_shards()
            self.assertEqual(len(shards), 7)
            task = _Task(iris_file_name, 0, 20, elasticai_api_pb2.TRAINING)
            record_count = 0
            for record in csv_data_reader.read_records(task):
                record_count += 1
            self.assertEqual(csv_data_reader.get_size(), num_records)
            self.assertEqual(record_count, 20)


@unittest.skipIf(
    not is_odps_configured(), "ODPS environment is not configured",
)
class ODPSDataReaderTest(unittest.TestCase):
    def setUp(self):
        self.project = os.environ[MaxComputeConfig.PROJECT_NAME]
        access_id = os.environ[MaxComputeConfig.ACCESS_ID]
        access_key = os.environ[MaxComputeConfig.ACCESS_KEY]
        endpoint = os.environ.get(MaxComputeConfig.ENDPOINT)
        tunnel_endpoint = os.environ.get(
            MaxComputeConfig.TUNNEL_ENDPOINT, None
        )
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
            tunnel_endpoint=tunnel_endpoint,
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
                _Task(
                    self.test_table + ":shard_0",
                    0,
                    2,
                    elasticai_api_pb2.TRAINING,
                )
            )
        )
        records = np.array(records, dtype="float").tolist()
        self.assertEqual(
            [[6.4, 2.8, 5.6, 2.2, 2], [5.0, 2.3, 3.3, 1.0, 1]], records
        )
        self.assertEqual(
            self.reader.metadata.column_names, IRIS_TABLE_COLUMN_NAMES
        )
        self.assertListEqual(
            list(self.reader.metadata.column_dtypes.values()),
            [
                odps.types.double,
                odps.types.double,
                odps.types.double,
                odps.types.double,
                odps.types.bigint,
            ],
        )
        self.assertEqual(
            self.reader.metadata.get_tf_dtype_from_maxcompute_column(
                self.reader.metadata.column_names[0]
            ),
            tf.float64,
        )
        self.assertEqual(
            self.reader.metadata.get_tf_dtype_from_maxcompute_column(
                self.reader.metadata.column_names[-1]
            ),
            tf.int64,
        )

    def test_create_data_reader(self):
        reader = create_data_reader(
            data_origin=self.test_table,
            records_per_task=10,
            **{
                "columns": ["sepal_length", "sepal_width"],
                "label_col": "class",
            }
        )
        self.assertEqual(
            reader._kwargs["columns"], ["sepal_length", "sepal_width"]
        )
        self.assertEqual(reader._kwargs["label_col"], "class")
        self.assertEqual(reader._kwargs["records_per_task"], 10)
        reader = create_data_reader(
            data_origin=self.test_table, records_per_task=10
        )
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
        reader = create_data_reader(
            data_origin=self.test_table,
            records_per_task=10,
            **{"columns": IRIS_TABLE_COLUMN_NAMES, "label_col": "class"}
        )
        feed = reader.default_feed()

        def _gen():
            for data in self.reader.read_records(
                _Task(
                    self.test_table + ":shard_0",
                    0,
                    num_records,
                    elasticai_api_pb2.TRAINING,
                )
            ):
                if data is not None:
                    yield data

        dataset = tf.data.Dataset.from_generator(_gen, tf.string)
        dataset = feed(
            dataset, None, Metadata(column_names=IRIS_TABLE_COLUMN_NAMES)
        )
        dataset = dataset.batch(1)

        loss_history = []
        grads = None
        for features, labels in dataset:
            with tf.GradientTape() as tape:
                logits = model(features, training=True)
                loss_value = loss(labels, logits)
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
