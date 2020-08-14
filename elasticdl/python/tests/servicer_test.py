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

import random
import unittest
from collections import defaultdict

import tensorflow as tf
from unittest.mock import Mock
from elasticdl.proto import elasticdl_pb2
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.master.task_dispatcher import _TaskDispatcher


def _get_variable_names(model_pb):
    return [v.name for v in model_pb.param]


class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__(name="test_model")
        self.dense_1 = tf.keras.layers.Dense(
            32, activation="relu", name="dense_1"
        )
        self.dense_2 = tf.keras.layers.Dense(
            1, activation="sigmoid", name="dense_2"
        )

    def call(self, inputs):
        x = self.dense_1(inputs)
        return self.dense_2(x)

    @staticmethod
    def input_shapes():
        return 10, 10

    @staticmethod
    def optimizer(lr=0.1):
        return tf.optimizers.SGD(lr)


class ServicerTest(unittest.TestCase):
    def setUp(self):
        self.master = Mock(
            task_d=None,
            instance_manager=None,
            distribution_strategy=None,
        )

    def test_get_empty_task(self):
        self.master.task_d = _TaskDispatcher(
            {}, {}, {}, records_per_task=3, num_epochs=2
        )
        master_servicer = MasterServicer(
            3,
            evaluation_service=None,
            master=self.master,
        )

        req = elasticdl_pb2.GetTaskRequest()

        # No task yet, make sure the returned versions are as expected.
        req.worker_id = 1
        task = master_servicer.get_task(req, None)
        self.assertEqual("", task.shard_name)
        self.assertEqual(0, task.model_version)

        master_servicer._version = 1
        task = master_servicer.get_task(req, None)
        self.assertEqual("", task.shard_name)
        self.assertEqual(1, task.model_version)

    def test_report_task_result(self):
        self.master.task_d = _TaskDispatcher(
            {"shard_1": (0, 10), "shard_2": (0, 9)},
            {},
            {},
            records_per_task=3,
            num_epochs=2,
        )
        master = MasterServicer(
            3, evaluation_service=None, master=self.master
        )

        # task to number of runs.
        tasks = defaultdict(int)
        while True:
            req = elasticdl_pb2.GetTaskRequest()
            req.worker_id = random.randint(1, 10)
            task = master.get_task(req, None)
            if not task.shard_name:
                break
            self.assertEqual(
                self.master.task_d._doing[task.task_id][0], req.worker_id
            )
            task_key = (task.shard_name, task.start, task.end)
            tasks[task_key] += 1
            report = elasticdl_pb2.ReportTaskResultRequest()
            report.task_id = task.task_id
            if task.start == 0 and tasks[task_key] == 1:
                # Simulate error reports.
                report.err_message = "Worker error"
            master.report_task_result(report, None)

        self.assertDictEqual(
            {
                ("shard_1", 0, 3): 3,
                ("shard_1", 3, 6): 2,
                ("shard_1", 6, 9): 2,
                ("shard_1", 9, 10): 2,
                ("shard_2", 0, 3): 3,
                ("shard_2", 3, 6): 2,
                ("shard_2", 6, 9): 2,
            },
            tasks,
        )


if __name__ == "__main__":
    unittest.main()
