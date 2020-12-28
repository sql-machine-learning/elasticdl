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

import unittest
from unittest.mock import MagicMock, Mock

from elasticai_api.common.data_shard_service import DataShardService
from elasticai_api.proto import elasticai_api_pb2
from elasticdl.python.master.task_manager import _Task


class DataShardServiceTest(unittest.TestCase):
    def setUp(self):
        self._master_client = Mock()
        self._master_client.get_task = MagicMock(
            return_value=_Task("test_file", 0, 1, elasticai_api_pb2.TRAINING)
        )
        self._master_client.report_task_result = MagicMock(return_value=True)

    def test_get_task(self):
        data_shard_service = DataShardService(self._master_client, 1)
        task = data_shard_service.get_task()
        self.assertEqual(task, data_shard_service.get_current_task())
        self.assertEqual(task.shard.start, 0)
        self.assertEqual(task.shard.end, 1)
        self.assertEqual(len(data_shard_service._pending_tasks), 1)

    def test_fetch_shard(self):
        data_shard_service = DataShardService(self._master_client, 1)
        shard = data_shard_service.fetch_shard()
        self.assertEqual(shard.name, "test_file")
        self.assertEqual(shard.start, 0)
        self.assertEqual(shard.end, 1)

    def test_report_batch_done(self):
        data_shard_service = DataShardService(self._master_client, 1)
        task = data_shard_service.get_task()
        task.task_id = 0
        reported = data_shard_service.report_batch_done()
        self.assertTrue(reported)
        self.assertEqual(len(data_shard_service._pending_tasks), 0)
