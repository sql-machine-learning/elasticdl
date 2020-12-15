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
import unittest

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.tests.test_utils import create_task_manager


class TaskManagerTest(unittest.TestCase):
    def test_create_tasks_with_zero_start_ind(self):
        task_d = create_task_manager([("f1", 0, 10), ("f2", 0, 10)], [])

        all_tasks = [
            ("f1", 0, 3, elasticdl_pb2.TRAINING, -1),
            ("f1", 3, 6, elasticdl_pb2.TRAINING, -1),
            ("f1", 6, 9, elasticdl_pb2.TRAINING, -1),
            ("f1", 9, 10, elasticdl_pb2.TRAINING, -1),
            ("f2", 0, 3, elasticdl_pb2.TRAINING, -1),
            ("f2", 3, 6, elasticdl_pb2.TRAINING, -1),
            ("f2", 6, 9, elasticdl_pb2.TRAINING, -1),
            ("f2", 9, 10, elasticdl_pb2.TRAINING, -1),
        ]

        # get all tasks out, each worker is assigned 2 tasks.
        got_tasks = [task_d.get(i // 2) for i in range(8)]

        # verify ids ranges from 1 to 8
        self.assertEqual(list(range(1, 9)), [k for k, _ in got_tasks])

        # verify tasks
        self.assertEqual(sorted([v._info() for _, v in got_tasks]), all_tasks)

        # no todo tasks, should return None
        self.assertEqual((-1, None), task_d.get(10))

        request = elasticdl_pb2.ReportTaskResultRequest()
        # report 6 task successes.
        for t in (1, 3, 5, 7, 2, 8):
            request.task_id = t
            task_d.report(request, True)

        # there should be 2 doing tasks left.
        self.assertEqual(2, len(task_d._doing))

        # report a task failure
        request.task_id = list(task_d._doing.items())[0][0]
        task_d.report(request, False)
        self.assertEqual(1, len(task_d._doing))

        # recover tasks from a dead worker
        task_d.recover_tasks(list(task_d._doing.items())[0][1][0])
        self.assertEqual(0, len(task_d._doing))

        self.assertEqual(2, len(task_d._todo))

        id1, t1 = task_d.get(11)
        id2, t2 = task_d.get(12)
        request.task_id = id1
        task_d.report(request, True)
        request.task_id = id2
        task_d.report(request, True)

        self.assertTrue(task_d.finished())

    def test_create_tasks_with_non_zero_start_ind(self):
        task_d = create_task_manager([("f1", 0, 10), ("f2", 10, 10)], [])

        all_tasks = [
            ("f1", 0, 3, elasticdl_pb2.TRAINING, -1),
            ("f1", 3, 6, elasticdl_pb2.TRAINING, -1),
            ("f1", 6, 9, elasticdl_pb2.TRAINING, -1),
            ("f1", 9, 10, elasticdl_pb2.TRAINING, -1),
            ("f2", 10, 13, elasticdl_pb2.TRAINING, -1),
            ("f2", 13, 16, elasticdl_pb2.TRAINING, -1),
            ("f2", 16, 19, elasticdl_pb2.TRAINING, -1),
            ("f2", 19, 20, elasticdl_pb2.TRAINING, -1),
        ]

        # get all tasks out, each worker is assigned 2 tasks.
        got_tasks = [task_d.get(i // 2) for i in range(8)]

        # verify ids ranges from 1 to 8
        self.assertEqual(list(range(1, 9)), [k for k, _ in got_tasks])

        # verify tasks
        self.assertEqual(sorted([v._info() for _, v in got_tasks]), all_tasks)

    def test_epoch(self):
        task_d = create_task_manager([("f1", 0, 10), ("f2", 0, 10)], [], 2)

        epoch_tasks = [
            ("f1", 0, 3, elasticdl_pb2.TRAINING, -1),
            ("f1", 3, 6, elasticdl_pb2.TRAINING, -1),
            ("f1", 6, 9, elasticdl_pb2.TRAINING, -1),
            ("f1", 9, 10, elasticdl_pb2.TRAINING, -1),
            ("f2", 0, 3, elasticdl_pb2.TRAINING, -1),
            ("f2", 3, 6, elasticdl_pb2.TRAINING, -1),
            ("f2", 6, 9, elasticdl_pb2.TRAINING, -1),
            ("f2", 9, 10, elasticdl_pb2.TRAINING, -1),
        ]

        # get first epoch tasks
        got_tasks = [task_d.get(i // 2) for i in range(8)]
        self.assertEqual(
            sorted([v._info() for _, v in got_tasks]), epoch_tasks
        )

        # get second epoch tasks
        got_tasks = [task_d.get(i // 2) for i in range(8)]
        self.assertEqual(
            sorted([v._info() for _, v in got_tasks]), epoch_tasks
        )

    def test_invoke_train_end_callback(self):
        task_d = create_task_manager([("f1", 0, 10), ("f2", 0, 10)], [])
        task_d._add_deferred_callback_create_train_end_task()
        task_d._todo.clear()
        task_d.invoke_deferred_callback()
        self.assertEqual(len(task_d._todo), 1)
        self.assertEqual(
            task_d._todo[0].type, elasticdl_pb2.TRAIN_END_CALLBACK
        )

    def test_check_and_reassign_timeout_tasks(self):
        task_manager = create_task_manager([("f1", 0, 10), ("f2", 0, 10)], [])
        task_manager.create_tasks(elasticdl_pb2.TRAINING)
        task_count = len(task_manager._todo)
        task_start_time = time.time() - 1000
        task_manager._worker_start_task_time[0] = task_start_time
        task_manager._doing[0] = (0, task_manager._todo[0], task_start_time)
        task_manager._todo.pop()

        threading.Thread(
            target=task_manager._check_and_reassign_timeout_tasks,
            name="check_timeout_tasks",
            daemon=True,
        ).start()
        time.sleep(1)  # Sleep 1s to reassgin checkout the timeout task
        self.assertEqual(len(task_manager._todo), task_count)

    def test_get_max_task_completed_time(self):
        task_manager = create_task_manager([("f1", 0, 10), ("f2", 0, 10)], [])
        self.assertEqual(
            task_manager._max_task_completed_times,
            {elasticdl_pb2.TRAINING: 0, elasticdl_pb2.EVALUATION: 0},
        )
        task_manager.record_task_completed_time(elasticdl_pb2.TRAINING, 10)
        task_manager.record_task_completed_time(elasticdl_pb2.EVALUATION, 5)

        self.assertEqual(
            task_manager._max_task_completed_times,
            {elasticdl_pb2.TRAINING: 10, elasticdl_pb2.EVALUATION: 5},
        )


if __name__ == "__main__":
    unittest.main()
