"""TaskQueue Implementation"""

import logging
import random
import threading
import time

from threading import Thread
from elasticdl.proto import elasticdl_pb2


class _Task(object):
    """Internal representation of a task"""

    def __init__(self, *, file_name, start, end, type, model_version=-1):
        self.file_name = file_name
        self.start = start
        self.end = end
        self.type = type
        self.model_version = model_version

    def _info(self):
        return (
            self.file_name,
            self.start,
            self.end,
            self.type,
            self.model_version,
        )


class _EvaluationJob(object):
    """Internal representation of an evaluation job"""

    def __init__(self, model_version, total_tasks=-1):
        self._model_version = model_version
        self._total_tasks = total_tasks
        self._completed_tasks = 0
        self._start_time = time.time()

    def complete_task(self):
        self._completed_tasks += 1

    def finished(self):
        return self._completed_tasks >= self._total_tasks

    def ok_to_new_job(self, throttle_secs, latest_chkp_version):
        return (
            self.finished()
            and latest_chkp_version > self._model_version
            and (time.time() - self._start_time) >= throttle_secs
        )


class _EvaluationTrigger(Thread):
    """A trigger which generates evaluation tasks periodically"""

    def __init__(
        self, master_servicer, task_q, stopped, start_delay_secs, throttle_secs
    ):
        Thread.__init__(self)
        self._master_servicer = master_servicer
        self._task_q = task_q
        self._stopped = stopped
        self._start_delay_secs = start_delay_secs
        self._throttle_secs = throttle_secs
        self._eval_job = None

    def run(self):
        time.sleep(self._start_delay_secs)
        while not self._stopped(1):
            latest_chkp_version = (
                self._master_servicer.get_last_checkpoint_version()
            )
            if self._eval_job is None or self._eval_job.ok_to_new_job(
                self._throttle_secs, latest_chkp_version
            ):
                self._eval_job = _EvaluationJob(latest_chkp_version)
                tasks = self._task_q.create_evaluation_tasks(
                    latest_chkp_version
                )
                self._eval_job._total_tasks = len(tasks)
                self._task_q.set_eval_job(self._eval_job)


class _TaskQueue(object):
    """Creates and dispatches Tasks. Keep track of a Task's lifecycle."""

    def __init__(
        self, training_shards, evaluation_shards, records_per_task, num_epochs
    ):
        """
        shards: a dictionary from RecordIO file name to number of records
        """
        self._logger = logging.getLogger(__name__)
        self._lock = threading.Lock()

        self._num_epochs = num_epochs
        self._epoch = 0
        self._training_shards = training_shards
        self._evaluation_shards = evaluation_shards
        self._records_per_task = records_per_task

        self._todo = []
        # dictionary from task id to Task.
        self._doing = {}
        self._task_id = 0
        self._eval_job = None

        self.create_training_tasks()

    def create_training_tasks(self):
        # with self._lock:
        self._logger.info(
            "Creating a new set of training tasks with epoch=%d" % self._epoch
        )
        tasks = self._create_tasks(
            self._training_shards, elasticdl_pb2.TRAINING
        )
        random.shuffle(tasks)
        self._todo.extend(tasks)
        return tasks

    def create_evaluation_tasks(self, eval_model_version):
        # with self._lock:
        self._logger.info(
            "Creating a new set of evaluation tasks for model version %d"
            % eval_model_version
        )
        tasks = self._create_tasks(
            self._evaluation_shards,
            elasticdl_pb2.EVALUATION,
            eval_model_version,
        )
        with self._lock:
            self._todo.extend(tasks)
        return tasks

    def _create_tasks(self, shards, task_type, model_version=-1):
        tasks = []
        for name, num_records in shards.items():
            for start in range(0, num_records, self._records_per_task):
                tasks.append(
                    _Task(
                        file_name=name,
                        start=start,
                        end=min(start + self._records_per_task, num_records),
                        type=task_type,
                        model_version=model_version,
                    )
                )
        return tasks

    def get(self, worker_id):
        """Return next (task_id, Task) tuple"""

        with self._lock:
            # TODO: check if task queue doesn't have training task,
            #       to avoid the queue is overwhelmed by evaluation tasks.
            if not self._todo and self._epoch < self._num_epochs - 1:
                # Start a new epoch
                self.create_training_tasks()
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
            print("acquire lock in report")
            _, task = self._doing.pop(task_id, (-1, None))
            if not task:
                self._logger.warning("Unknown task_id: %d" % task_id)
            elif not success:
                # TODO: keep count of retries.
                self._todo.append(task)
            elif (
                task.type == elasticdl_pb2.EVALUATION
                and self._eval_job is not None
            ):
                self._eval_job.complete_task()

    def finished(self):
        """Return if all tasks are done"""
        return not self._todo and not self._doing

    def recover_tasks(self, worker_id):
        """Recover doing tasks for a dead worker"""

        with self._lock:
            print("acquire lock in recover")
            ids = [
                id for id, (wid, _) in self._doing.items() if wid == worker_id
            ]
        for id in ids:
            self.report(id, False)

    # TODO: need to re-check after refactoring servicer.py
    def set_eval_job(self, eval_job):
        with self._lock:
            print("acquire lock in set_eval")
            self._eval_job = eval_job
