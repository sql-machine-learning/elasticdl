"""TaskQueue Implementation"""

import logging
import random
import threading

from elasticdl.proto import elasticdl_pb2


class _Task(object):
    """Internal representation of a task"""

    def __init__(self, *, file_name, start, end, type):
        self.file_name = file_name
        self.start = start
        self.end = end
        self.type = type

    def _info(self):
        return self.file_name, self.start, self.end, self.type


class _TaskQueue(object):
    """Creates and dispatches Tasks. Keep track of a Task's lifecycle."""

    def __init__(self, training_shards, evaluation_shards, record_per_task, num_epoch, num_evaluation_epoch):
        """
        shards: a dictionary from RecordIO file name to number of records
        """
        self._logger = logging.getLogger(__name__)
        self._lock = threading.Lock()

        self._num_epoch = num_epoch
        self._num_evaluation_epoch = num_evaluation_epoch
        self._epoch = 0
        self._training_shards = training_shards
        self._evaluation_shards = evaluation_shards
        self._record_per_task = record_per_task

        self._todo = []
        # dictionary from task id to Task.
        self._doing = {}
        self._task_id = 0

        self._todo.extend(self._create_training_tasks())

    def _create_training_tasks(self):
        training_todo = []
        for name, num_records in self._training_shards.items():
            for start in range(0, num_records, self._record_per_task):
                training_todo.append(
                    _Task(
                        file_name=name,
                        start=start,
                        end=min(start + self._record_per_task, num_records),
                        type=elasticdl_pb2.TRAINING,
                    )
                )
        random.shuffle(training_todo)
        return training_todo

    def _create_evaluation_tasks(self):
        evaluation_todo = []
        for name, num_records in self._evaluation_shards.items():
            for start in range(0, num_records, self._record_per_task):
                evaluation_todo.append(
                    _Task(
                        file_name=name,
                        start=start,
                        end=min(start + self._record_per_task, num_records),
                        type=elasticdl_pb2.EVALUATION,
                    )
                )
        random.shuffle(evaluation_todo)
        return evaluation_todo

    def _need_evaluation(self):
        return self._epoch != 0 and self._epoch % self._num_evaluation_epoch == 0

    def get(self, worker_id):
        """Return next (task_id, Task) tuple"""

        with self._lock:
            if not self._todo and self._epoch < self._num_epoch - 1:
                # Start a new epoch
                self._logger.info("Generating training tasks")
                training_todo = self._create_training_tasks()
                self._todo.extend(training_todo)
                if self._need_evaluation():
                    self._logger.info("Generating evaluation tasks")
                    evaluation_todo = self._create_evaluation_tasks()
                    self._todo.extend(evaluation_todo)
                self._epoch += 1
                self._logger.info("Starting epoch %d" % self._epoch)

            if not self._todo:
                # No more tasks
                return -1, None

            self._task_id += 1
            task = self._todo.pop()
            # TODO: Handle timeout of tasks.
            self._doing[self._task_id] = (worker_id, task)

            return self._task_id, task

    def report(self, task_id, success):
        """Report if the task is successful or not"""

        with self._lock:
            _, task = self._doing.pop(task_id, (-1, None))
            if not task:
                self._logger.warning("Unknown task_id: %d" % task_id)
            elif not success:
                # TODO: keep count of retries.
                # 这里就会有问题，如果放在最后面的话，就会导致evaluation和training的不一致
                self._todo.insert(0, task)

    def finished(self):
        """Return if all tasks are done"""
        return not self._todo and not self._doing

    def recover_tasks(self, worker_id):
        """Recover doing tasks for a dead worker"""

        with self._lock:
            ids = [
                id for id, (wid, _) in self._doing.items() if wid == worker_id
            ]
        for id in ids:
            self.report(id, False)
