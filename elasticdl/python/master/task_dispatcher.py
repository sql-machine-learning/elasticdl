"""TaskQueue Implementation"""

import random
import threading

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.log_util import default_logger as logger


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


class _TaskDispatcher(object):
    """Creates and dispatches Tasks. Keep track of a Task's lifecycle."""

    def __init__(
        self,
        training_shards,
        evaluation_shards,
        prediction_shards,
        records_per_task,
        num_epochs,
    ):
        """
        Arguments:
            training_shards: A dictionary from RecordIO file name to the
                number of training records.
            evaluation_shards: A dictionary from RecordIO file name to
                the number of evaluation records.
            prediction_shards: A dictionary from RecordIO file name to
                the number of prediction records.
            records_per_task: The number of records per task.
            num_epochs: The total number of epochs for the tasks where
                an epoch is a complete iteration over the shards.
        """
        self._lock = threading.Lock()

        self._num_epochs = num_epochs
        self._epoch = 0
        self._training_shards = training_shards
        self._evaluation_shards = evaluation_shards
        self._prediction_shards = prediction_shards
        self._records_per_task = records_per_task

        self._todo = []
        # dictionary from task id to Task.
        self._doing = {}
        self._task_id = 0
        self._evaluation_service = None

        if self._training_shards:
            logger.info("Starting epoch %d", self._epoch)
            self.create_training_tasks()
        elif self._evaluation_shards:
            self.create_evaluation_tasks(-1)
        elif self._prediction_shards:
            self.create_prediction_tasks(-1)

    def create_training_tasks(self):
        logger.info(
            "Creating a new set of training tasks with epoch=%d", self._epoch
        )
        tasks = self._create_tasks(
            self._training_shards, elasticdl_pb2.TRAINING
        )
        random.shuffle(tasks)
        self._todo.extend(tasks)
        return tasks

    def create_evaluation_tasks(self, eval_model_version):
        logger.info(
            "Creating a new set of evaluation tasks for model version %d",
            eval_model_version,
        )
        tasks = self._create_tasks(
            self._evaluation_shards,
            elasticdl_pb2.EVALUATION,
            eval_model_version,
        )
        with self._lock:
            self._todo.extend(tasks)
        return tasks

    def create_prediction_tasks(self, predict_model_version):
        logger.info(
            "Creating a new set of prediction tasks for model version %d",
            predict_model_version,
        )
        tasks = self._create_tasks(
            self._prediction_shards,
            elasticdl_pb2.PREDICTION,
            predict_model_version,
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
                self._epoch += 1
                self.create_training_tasks()
                logger.info("Starting epoch %d", self._epoch)

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

        evaluation_task_completed = False
        with self._lock:
            _, task = self._doing.pop(task_id, (-1, None))
            if not task:
                logger.warning("Unknown task_id: %d" % task_id)
            elif not success:
                # TODO: keep count of retries.
                self._todo.append(task)
            elif (
                task.type == elasticdl_pb2.EVALUATION
                and self._evaluation_service is not None
            ):
                evaluation_task_completed = True
            else:
                logger.info(
                    "Task:%d completed, %d remaining tasks",
                    task_id,
                    len(self._todo) + len(self._doing),
                )
        if evaluation_task_completed:
            self._evaluation_service.complete_task()

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

    # TODO: need to re-check after refactoring servicer.py
    def set_evaluation_service(self, evaluation_service):
        with self._lock:
            self._evaluation_service = evaluation_service
            if self._evaluation_shards and not self._training_shards:
                evaluation_service.init_eval_only_job(len(self._todo))
