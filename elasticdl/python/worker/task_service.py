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
import threading
from collections import deque

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.constants import TaskExecCounterKey
from elasticdl.python.common.grpc_utils import build_channel
from elasticdl.python.worker.master_client import MasterClient


class TaskService(object):
    def __init__(
        self, master_client=None,
    ):
        if master_client is None:
            master_addr = os.getenv("MASTER_ADDR")
            worker_id = int(os.getenv("WORKER_ID"))
            self._mc = MasterClient(build_channel(master_addr), worker_id)
        else:
            self._mc = master_client

        self._lock = threading.Lock()
        self._failed_record_count = 0
        self._reported_record_count = 0
        self._current_task = None
        self._pending_tasks = deque()

    def get_current_task(self):
        return self._current_task

    def get_task(self, task_type=None):
        task = self._mc.get_task(task_type)
        if (
            task.shard.name
            and task.type == elasticdl_pb2.TRAINING
        ):
            self._pending_tasks.append(task)
            if len(self._pending_tasks) == 1:
                self._current_task = task
        return task

    def fetch_shard(self):
        task = self.get_task()
        return task.shard

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

    def report_batch_done(self, count, err_msg=""):
        """
        Report the number of records in the latest processed batch,
        so DynamicShardingManager knows if some pending tasks are finished
        and report_task_result to the master.
        Return True if there are some finished tasks, False otherwise.
        """
        self._reported_record_count += count
        if err_msg:
            self._failed_record_count += count

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
                    self._pending_tasks.popleft()
                    self._report_task(task, err_msg)
                    self._failed_record_count = 0
                if self._pending_tasks:
                    self._current_task = self._pending_tasks[0]
            return True
        return False
