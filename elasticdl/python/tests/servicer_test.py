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
from unittest.mock import MagicMock, Mock

import tensorflow as tf

from elasticai_api.proto import elasticai_api_pb2
from elasticdl.python.master.rendezvous_server import HorovodRendezvousServer
from elasticdl.python.master.servicer import (
    MasterServicer,
    create_master_service,
)
from elasticdl.python.tests.test_utils import create_task_manager


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
            task_d=None, instance_manager=None, distribution_strategy=None,
        )

    def test_create_master_service(self):
        server = create_master_service(8080, None, None, None, None)
        self.assertIsNotNone(server)

    def test_get_empty_task(self):
        self.master.task_manager = create_task_manager([], [])
        master_servicer = MasterServicer(
            self.master.task_manager, self.master.instance_manager, None, None,
        )

        req = elasticai_api_pb2.GetTaskRequest()

        # No task yet, make sure the returned versions are as expected.
        req.worker_id = 1
        task = master_servicer.get_task(req, None)
        self.assertEqual("", task.shard.name)
        self.assertEqual(0, task.model_version)

        master_servicer._version = 1
        task = master_servicer.get_task(req, None)
        self.assertEqual("", task.shard.name)
        self.assertEqual(1, task.model_version)

    def test_report_task_result(self):
        self.master.task_manager = create_task_manager(
            [("shard_1", 0, 10), ("shard_2", 0, 9)], [], 2
        )
        master = MasterServicer(
            self.master.task_manager, self.master.instance_manager, None, None,
        )

        # task to number of runs.
        tasks = defaultdict(int)
        while True:
            req = elasticai_api_pb2.GetTaskRequest()
            req.worker_id = random.randint(1, 10)
            task = master.get_task(req, None)
            if not task.shard.name:
                break
            self.assertEqual(
                self.master.task_manager._doing[task.task_id][0], req.worker_id
            )
            task_key = (task.shard.name, task.shard.start, task.shard.end)
            tasks[task_key] += 1
            report = elasticai_api_pb2.ReportTaskResultRequest()
            report.task_id = task.task_id
            if task.shard.start == 0 and tasks[task_key] == 1:
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

    def test_get_comm_rank(self):
        self.master.rendezvous_server = HorovodRendezvousServer(
            server_host="localhost"
        )
        self.master.rendezvous_server.start()
        self.master.rendezvous_server.add_worker("172.0.0.1")
        self.master.rendezvous_server.add_worker("172.0.0.2")

        mock_instance_manager = Mock()
        mock_instance_manager.get_worker_pod_ip = MagicMock(
            return_value="172.0.0.1"
        )
        self.master.instance_manager = mock_instance_manager
        master_servicer = MasterServicer(
            self.master.task_manager,
            self.master.instance_manager,
            self.master.rendezvous_server,
            None,
        )
        request = elasticai_api_pb2.GetCommRankRequest()
        request.worker_host = "172.0.0.1"
        rank_response = master_servicer.get_comm_rank(request, None)
        self.assertEqual(rank_response.world_size, 2)
        self.assertEqual(rank_response.rank_id, 0)
        self.assertEqual(rank_response.rendezvous_id, 1)


if __name__ == "__main__":
    unittest.main()
