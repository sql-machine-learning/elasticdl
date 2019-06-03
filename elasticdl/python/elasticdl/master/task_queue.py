"""TaskQueue Implementation"""

import logging
import random
import threading


class _Task(object):
    """Internal representation of a task"""

    def __init__(self, *, file_name, start, end):
        self.file_name = file_name
        self.start = start
        self.end = end

    def _info(self):
        return self.file_name, self.start, self.end


class _TaskQueue(object):
    """Creates and dispatches Tasks. Keep track of a Task's lifecycle."""

    def __init__(self, shards, record_per_task, num_epoch):
        """
        shards: a dictionary from RecordIO file name to number of records
        """
        self._logger = logging.getLogger(__name__)
        self._lock = threading.Lock()

        self._num_epoch = num_epoch
        self._epoch = 0
        self._shards = shards
        self._record_per_task = record_per_task

        self._todo = []
        # dictionary from task id to Task.
        self._doing = {}
        self._task_id = 0

        self._create_tasks()

    def _create_tasks(self):
        for name, num_records in self._shards.items():
            for start in range(0, num_records, self._record_per_task):
                self._todo.append(
                    _Task(
                        file_name=name,
                        start=start,
                        end=min(start + self._record_per_task, num_records),
                    )
                )
        random.shuffle(self._todo)

    def get(self, worker_id):
        """Return next (task_id, Task) tuple"""

        with self._lock:
            if not self._todo and self._epoch < self._num_epoch - 1:
                # Start a new epoch
                self._create_tasks()
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
                self._todo.append(task)

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
