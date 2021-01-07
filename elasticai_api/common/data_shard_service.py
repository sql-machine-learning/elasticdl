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

import threading
import time
from collections import deque
from multiprocessing import SimpleQueue

from elasticai_api.common.constants import TaskExecCounterKey
from elasticai_api.common.master_client import build_master_client
from elasticai_api.proto import elasticai_api_pb2


def build_data_shard_service(
    batch_size,
    num_epochs=None,
    dataset_size=None,
    shuffle=False,
    shuffle_shards=False,
    task_type=elasticai_api_pb2.TRAINING,
):
    master_client = build_master_client()
    return DataShardService(
        batch_size=batch_size,
        master_client=master_client,
        num_epochs=num_epochs,
        dataset_size=dataset_size,
        shuffle=shuffle,
        shuffle_shards=shuffle_shards,
        task_type=task_type,
    )


class DataShardService(object):
    def __init__(
        self,
        master_client,
        batch_size,
        num_epochs=None,
        dataset_size=None,
        shuffle=False,
        shuffle_shards=False,
        task_type=elasticai_api_pb2.TRAINING,
    ):
        self._mc = master_client
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._dataset_size = dataset_size
        self._shuffle = shuffle
        self._shuffle_shards = shuffle_shards
        self._task_type = task_type
        self._lock = threading.Lock()
        self._failed_record_count = 0
        self._reported_record_count = 0
        self._current_task = None
        self._pending_tasks = deque()
        self._report_training_params()

    def _report_training_params(self):
        if self._num_epochs and self._dataset_size:
            self._mc.report_training_params(
                batch_size=self._batch_size,
                num_epochs=self._num_epochs,
                dataset_size=self._dataset_size,
                shuffle=self._shuffle,
                shuffle_shards=self._shuffle_shards,
            )

    def get_current_task(self):
        return self._current_task

    def get_task(self, task_type=None):
        task = self._mc.get_task(task_type)
        if task.type == self._task_type:
            with self._lock:
                self._pending_tasks.append(task)
                if len(self._pending_tasks) == 1:
                    self._current_task = task

        return task

    def _report_task(self, task, err_msg=""):
        if self._failed_record_count != 0:
            exec_counters = {
                TaskExecCounterKey.FAIL_COUNT: self._failed_record_count
            }
        else:
            exec_counters = None
        self._mc.report_task_result(
            task.task_id, err_msg, exec_counters=exec_counters
        )

    def report_batch_done(self, batch_size=None, err_msg=""):
        """
        Report the number of records in the latest processed batch,
        so DynamicShardingManager knows if some pending tasks are finished
        and report_task_result to the master.
        Return True if there are some finished tasks, False otherwise.
        """
        record_count = batch_size if batch_size else self._batch_size
        self._reported_record_count += record_count
        if err_msg:
            self._failed_record_count += record_count

        if not self._pending_tasks:
            return False
        task = self._pending_tasks[0]
        total_record_num = task.shard.end - task.shard.start
        if self._reported_record_count >= total_record_num:
            # Keep popping tasks until the reported record count is less
            # than the size of the current data since `batch_size` may be
            # larger than `shard.end - shard.start`
            with self._lock:
                while (
                    self._pending_tasks
                    and self._reported_record_count
                    >= self._pending_tasks[0].shard.end
                    - self._pending_tasks[0].shard.start
                ):
                    self._reported_record_count -= (
                        self._pending_tasks[0].shard.end
                        - self._pending_tasks[0].shard.start
                    )
                    task = self._pending_tasks.popleft()
                    self._report_task(task, err_msg)
                    self._failed_record_count = 0
                if self._pending_tasks:
                    self._current_task = self._pending_tasks[0]
            return True
        return False

    def fetch_shard(self):
        """Fetch data shard and each shard contains the name,
        start and end index.
        """
        task = self.get_task(self._task_type)
        if task.type != self._task_type:
            return None

        return task.shard


class RecordIndexService(DataShardService):
    def __init__(
        self,
        master_client,
        batch_size,
        num_epochs=None,
        dataset_size=None,
        task_type=elasticai_api_pb2.TRAINING,
        shuffle=False,
    ):
        super(RecordIndexService, self).__init__(
            master_client=master_client,
            batch_size=batch_size,
            num_epochs=num_epochs,
            dataset_size=dataset_size,
            shuffle=shuffle,
            task_type=task_type,
        )
        self._shard_queue = SimpleQueue()
        threading.Thread(
            target=self._get_shard_indices,
            name="fetch_shard_indices",
            daemon=True,
        ).start()

    def _get_shard_indices(self):
        while True:
            if self._shard_queue.empty():
                task = self.get_task(self._task_type)
                if not task.shard or task.type != self._task_type:
                    break
                ids = (
                    task.shard.indices
                    if task.shard.indices
                    else list(range(task.shard.start, task.shard.end))
                )
                for i in ids:
                    self._shard_queue.put(i)
            else:
                time.sleep(1)

    def fetch_record_index(self):
        """Fetch an index of the record. The function get an index
        from a queue because there may be multiple sub-process to call
        the function.
        """
        for _ in range(30):
            if not self._shard_queue.empty():
                return self._shard_queue.get()
            else:
                time.sleep(1)
        raise StopIteration
