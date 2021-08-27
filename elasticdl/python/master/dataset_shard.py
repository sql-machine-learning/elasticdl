# Copyright 2021 The ElasticDL Authors. All rights reserved.
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

import json
import math
import random
import time

from elasticai_api.common.constants import DefaultDatasetName
from elasticai_api.proto import elasticai_api_pb2
from elasticdl.python.common.log_utils import default_logger as logger

_CHECKPOINT_VERSION = "v1.0"
_MAX_TASK_RETRIES = 3
_MAX_SHARD_COUNT = 50000


def create_shards_by_range(start_idx, end_idx, records_per_task):
    shards = []
    num_shards = (end_idx - start_idx) // records_per_task
    for _ in range(num_shards):
        shards.append(("", start_idx, records_per_task,))
        start_idx += records_per_task
    # Create a shard with the last records
    num_records_left = (end_idx - start_idx) % records_per_task
    if num_records_left != 0:
        shards.append(("", start_idx, num_records_left,))
    logger.info("In func:Create {} shards".format(len(shards)))
    return shards


class Shard(object):
    def __init__(self, name, start, end, indices=None):
        self.name = name
        self.start = start
        self.end = end
        self.indices = indices


class Task(object):
    """Internal representation of a task"""

    def __init__(
        self,
        shard_name,
        start,
        end,
        type,
        model_version=-1,
        task_record_indices=None,
        **kwargs
    ):
        self.shard = Shard(shard_name, start, end, task_record_indices)
        self.type = type
        self.model_version = model_version
        self.extended_config = kwargs

    def _info(self):
        return (
            self.shard.name,
            self.shard.start,
            self.shard.end,
            self.type,
            self.model_version,
        )


class ShardCheckpoint(object):
    def __init__(
        self,
        dataset_name,
        todo,
        doing,
        current_epoch,
        num_epochs,
        records_per_task,
        dataset_size,
        shuffle_shards,
        version,
        current_subepoch=0,
        **kwargs
    ):
        """
        Args:
            todo: [[start_0, end_0], [start_1, end_1]],
            doing: [[start_2, end_2], [start_3, end_3]],
            current_epoch: int64, the index of epoch,
            num_epochs: int64, the number of epoch,
            records_per_task: int64, the number of records per task,
            dataset_size: int64, the size of dataset,
            shuffle_shards: bool, true of false.
            version: string, check whether
                the version of checkpoint is valid.
            current_subepoch: int64, the index of subepoch,
            maybe it doesn't exist,
        """

        self.dataset_name = dataset_name
        self.todo = todo
        self.doing = doing
        self.current_epoch = current_epoch
        self.num_epochs = num_epochs
        self.records_per_task = records_per_task
        self.dataset_size = dataset_size
        self.shuffle_shards = shuffle_shards
        self.version = version
        self.current_subepoch = current_subepoch

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, checkpoint_str):
        checkpoint_dict = json.loads(checkpoint_str)
        return ShardCheckpoint(**checkpoint_dict)


class Dataset(object):
    def __init__(
        self,
        shuffle_shards,
        records_per_task,
        dataset_size,
        num_epochs,
        dataset_name,
        max_shard_count=_MAX_SHARD_COUNT,
    ):
        self._num_epochs = num_epochs
        self._dataset_name = dataset_name
        self._shuffle_shards = shuffle_shards
        self._records_per_task = records_per_task
        self._shards = []
        self.reset()
        self._max_shard_count = max_shard_count
        self._subepoch_idx = 0
        self._epoch = 0
        if dataset_size:
            self._dataset_size = dataset_size
        else:
            self._dataset_size = 0
        self._subepoch_num_per_epoch = 0

    def set_shards(self, shards):
        self._shards = shards

    def set_epoch(self, epoch):
        self._epoch = epoch

    def get_epoch(self):
        return self._epoch

    def get_shards(self):
        return self._shards

    def reset(self):
        self.todo = []
        self.doing = {}
        self._epoch = 0
        self._task_id = 0
        self._task_retry_count = {}
        self.max_task_completed_time = 0
        self._shards = []
        self._subepoch_idx = 0

    def get_task(self, worker_id, support_fault_tolerance):
        """Return next (task_id, Task) tuple"""

        if (
            not self.todo
            and self._dataset_name != DefaultDatasetName.EVALUATION
        ):
            # Start a new epoch
            # num_epochs <= 0 indicates that the master will create data
            # shards infinitely. So, the worker can use the dataset like
            # `dataset.repeat()`.
            if self._num_epochs <= 0 or self._epoch < self._num_epochs:
                self.create_tasks()
            elif (
                self._epoch == self._num_epochs
                and self._subepoch_idx < self._subepoch_num_per_epoch
            ):
                self.create_tasks()
        if not self.todo:
            # No more tasks
            return -1, None

        self._task_id += 1
        task = self.todo.pop(0)
        if support_fault_tolerance:
            self.doing[self._task_id] = (worker_id, task, time.time())

        return self._task_id, task

    def completed(self):
        return not self.todo and not self.doing

    def create_tasks(self, model_version=-1):
        logger.info(
            "Creating a new set of tasks for dataset {} with epoch {}".format(
                self._dataset_name, self._epoch
            )
        )
        tasks = []
        shard_count = math.ceil(self._dataset_size / self._records_per_task)
        total_records_count = self._dataset_size
        if shard_count <= self._max_shard_count:
            self._epoch += 1
            if not self._shards:
                self._shards = create_shards_by_range(
                    0, self._dataset_size, self._records_per_task
                )
        else:
            self._subepoch_num_per_epoch = math.ceil(
                shard_count / self._max_shard_count
            )
            if self._subepoch_idx >= self._subepoch_num_per_epoch:
                self._subepoch_idx = 0

            if self._subepoch_idx == 0:
                self._epoch += 1

            self._subepoch_idx += 1

            logger.info(
                "Creating tasks for dataset:{} in a subepoch, "
                "subepoch_idx:{}, subepoch_num:{}, epoch:{}".format(
                    self._dataset_name,
                    self._subepoch_idx,
                    self._subepoch_num_per_epoch,
                    self._epoch,
                )
            )

            subepoch_records = self._max_shard_count * self._records_per_task
            start_idx = (self._subepoch_idx - 1) * subepoch_records
            end_idx = start_idx + subepoch_records
            if end_idx > self._dataset_size:
                end_idx = self._dataset_size

            self._shards = create_shards_by_range(
                start_idx, end_idx, self._records_per_task
            )
            total_records_count = end_idx - start_idx
        tasks = self.create_tasks_with_shards(
            self._shards, self._records_per_task, self._shuffle_shards,
        )

        self.todo.extend(tasks)
        logger.info(
            "todo.extend: %d tasks created for "
            "dataset = %s with total of %s records."
            % (len(tasks), self._dataset_name, total_records_count)
        )

    def create_tasks_with_shards(
        self, shards, len_per_task, shuffle_shards, model_version=-1
    ):
        tasks = []
        for (
            shard_name,
            start_idx_this_shard,
            num_records_this_shard,
        ) in shards:
            max_idx_this_shard = start_idx_this_shard + num_records_this_shard
            for start_idx_this_task in range(
                start_idx_this_shard, max_idx_this_shard, len_per_task,
            ):
                end_idx_this_task = min(
                    start_idx_this_task + len_per_task, max_idx_this_shard,
                )

                # Note that only records in [start, end) of this task
                # will be consumed later in the worker that handles
                # this task.
                tasks.append(
                    Task(
                        shard_name=shard_name,
                        start=start_idx_this_task,
                        end=end_idx_this_task,
                        type=elasticai_api_pb2.TRAINING,
                        model_version=model_version,
                    )
                )
        if shuffle_shards:
            random.shuffle(tasks)
        return tasks

    def recover_task(self, task):
        if not self._check_exceed_max_task_retries(task):
            self.todo.append(task)

    def _check_exceed_max_task_retries(self, task):
        self._task_retry_count.setdefault(task, 1)
        self._task_retry_count[task] += 1
        if self._task_retry_count[task] > _MAX_TASK_RETRIES:
            logger.error(
                "A task %s of failed with %d retries "
                % (self._dataset_name, _MAX_TASK_RETRIES)
            )
            return True
        return False

    def get_checkpoint(self):
        todo_shards = []
        for task in self.todo:
            todo_shards.append([task.shard.start, task.shard.end])

        doing_shards = []
        for task_id in self.doing:
            task = self.doing[task_id][1]
            doing_shards.append([task.shard.start, task.shard.end])

        return ShardCheckpoint(
            dataset_name=self._dataset_name,
            todo=todo_shards,
            doing=doing_shards,
            current_epoch=self._epoch,
            num_epochs=self._num_epochs,
            records_per_task=self._records_per_task,
            dataset_size=self._dataset_size,
            shuffle_shards=self._shuffle_shards,
            version=_CHECKPOINT_VERSION,
            current_subepoch=self._subepoch_idx,
        )

    @classmethod
    def restore_from_shard_checkpoint(cls, shard_checkpoint):
        dataset = Dataset(
            shuffle_shards=shard_checkpoint.shuffle_shards,
            records_per_task=shard_checkpoint.records_per_task,
            dataset_size=shard_checkpoint.dataset_size,
            num_epochs=shard_checkpoint.num_epochs,
            dataset_name=shard_checkpoint.dataset_name,
        )
        dataset.set_epoch(shard_checkpoint.current_epoch)
        for shard_indices in shard_checkpoint.todo:
            dataset.todo.append(
                Task(
                    "",
                    shard_indices[0],
                    shard_indices[1],
                    elasticai_api_pb2.TRAINING,
                )
            )

        for shard_indices in shard_checkpoint.doing:
            dataset.todo.append(
                Task(
                    "",
                    shard_indices[0],
                    shard_indices[1],
                    elasticai_api_pb2.TRAINING,
                )
            )
        return dataset
