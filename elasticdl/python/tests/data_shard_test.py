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

import math
import unittest

from elasticai_api.proto import elasticai_api_pb2
from elasticdl.python.master.dataset_shard import Dataset


class DatasetShardSubepochTest(unittest.TestCase):
    def test_get_task(self):
        shuffle_shards = False
        records_per_task = 2
        dataset_size = 49
        num_epochs = 3
        max_shard_count = 100
        dataset_name = "test_get_task"
        dataset = Dataset(
            shuffle_shards,
            records_per_task,
            dataset_size,
            num_epochs,
            dataset_name,
            max_shard_count,
        )
        shard_per_epoch = math.ceil(dataset_size / records_per_task)
        index = 0
        while True:
            task_id, task = dataset.get_task(0, True)
            if task_id == -1:
                break
            self.assertEqual(task_id, index + 1)
            self.assertEqual(
                task.shard.start, (index % shard_per_epoch) * records_per_task
            )
            expected_end = (
                index % shard_per_epoch
            ) * records_per_task + records_per_task
            expected_end = (
                expected_end if expected_end <= dataset_size else dataset_size
            )
            self.assertEqual(task.shard.end, expected_end)
            index += 1
            self.assertEqual(
                dataset._epoch, math.ceil(index / shard_per_epoch)
            )
        self.assertEqual(index, shard_per_epoch * num_epochs)

    def test_get_task_with_subepoch(self):
        shuffle_shards = False
        records_per_task = 2
        dataset_size = 49
        num_epochs = 3
        max_shard_count = 10
        dataset_name = "test_get_task_with_subepoch"
        dataset = Dataset(
            shuffle_shards,
            records_per_task,
            dataset_size,
            num_epochs,
            dataset_name,
            max_shard_count,
        )
        shard_per_epoch = math.ceil(dataset_size / records_per_task)
        index = 0
        while True:
            task_id, task = dataset.get_task(0, True)
            if task_id == -1:
                break
            self.assertEqual(task_id, index + 1)
            self.assertEqual(
                task.shard.start, (index % shard_per_epoch) * records_per_task
            )
            expected_end = (
                index % shard_per_epoch
            ) * records_per_task + records_per_task
            expected_end = (
                expected_end if expected_end <= dataset_size else dataset_size
            )
            self.assertEqual(task.shard.end, expected_end)
            index += 1
            self.assertEqual(
                dataset._epoch, math.ceil(index / shard_per_epoch)
            )
            shard_index_in_epoch = index - shard_per_epoch * (
                dataset._epoch - 1
            )
            self.assertEqual(
                dataset._subepoch_idx,
                math.ceil(shard_index_in_epoch / max_shard_count),
            )

        self.assertEqual(index, shard_per_epoch * num_epochs)

    def test_dataset(self):
        dataset = Dataset(False, 100, 10000, 2, "test_data_0")
        for i in range(2):
            task_id, task = dataset.get_task(0, True)
            self.assertEqual(task_id, i + 1)
            self.assertEqual(task.type, elasticai_api_pb2.TRAINING)
            self.assertEqual(len(dataset.doing), i + 1)
            self.assertEqual(task.shard.start, i * 100)
            self.assertEqual(task.shard.end, i * 100 + 100)
        dataset.reset()
        for i in range(2):
            task_id, task = dataset.get_task(0, True)
            self.assertEqual(task_id, i + 1)
            self.assertEqual(task.type, elasticai_api_pb2.TRAINING)
            self.assertEqual(len(dataset.doing), i + 1)
            self.assertEqual(task.shard.start, i * 100)
            self.assertEqual(task.shard.end, i * 100 + 100)


if __name__ == "__main__":
    unittest.main()
