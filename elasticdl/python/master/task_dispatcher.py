"""TaskQueue Implementation"""

import random
import threading

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.log_utils import default_logger as logger


class _Task(object):
    """Internal representation of a task"""

    def __init__(self, *, shard_name, start, end, type, model_version=-1):
        self.shard_name = shard_name
        self.start = start
        self.end = end
        self.type = type
        self.model_version = model_version

    def _info(self):
        return (
            self.shard_name,
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
        need_save_model=False,
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
            need_save_model: Whether to save model after all tasks
                are completed.
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

        # Callback list to invoke after all tasks complete.
        self._task_list_done_callbacks = []
        if need_save_model:
            self._task_list_done_callbacks.append(self._create_save_model_task)

        if self._training_shards:
            logger.info("Starting epoch %d", self._epoch)
            self.create_tasks(elasticdl_pb2.TRAINING)
        elif self._evaluation_shards:
            self.create_tasks(elasticdl_pb2.EVALUATION)
        elif self._prediction_shards:
            self.create_tasks(elasticdl_pb2.PREDICTION)

    def create_tasks(self, task_type, model_version=-1):
        logger.info(
            "Creating a new set of %s tasks for model version %d",
            elasticdl_pb2._TASKTYPE.values_by_number[task_type].name.lower(),
            model_version,
        )
        if task_type == elasticdl_pb2.TRAINING:
            shards = self._training_shards
        elif task_type == elasticdl_pb2.EVALUATION:
            shards = self._evaluation_shards
        else:
            shards = self._prediction_shards
        tasks = []
        # Note that a shard may contain records for multiple tasks.
        for (
            shard_name,
            (start_ind_this_shard, num_records_this_shard),
        ) in shards.items():
            max_ind_this_shard = start_ind_this_shard + num_records_this_shard
            for start_ind_this_task in range(
                start_ind_this_shard,
                start_ind_this_shard + num_records_this_shard,
                self._records_per_task,
            ):
                max_ind_this_task = (
                    start_ind_this_task + self._records_per_task
                )
                end_ind_this_task = min(
                    max_ind_this_task, num_records_this_shard
                )
                # If the start index is not smaller than end index,
                # we need to find the correct end index by taking the start
                # index into account. We should not create task with
                # end index that exceeds the maximally possible number of
                # records available in this shard.
                if start_ind_this_task >= end_ind_this_task:
                    end_ind_this_task = min(
                        max_ind_this_task,
                        start_ind_this_task + num_records_this_shard,
                        max_ind_this_shard,
                    )
                # Note that only records in [start, end) of this task
                # will be consumed later in the worker that handles
                # this task.
                tasks.append(
                    _Task(
                        shard_name=shard_name,
                        start=start_ind_this_task,
                        end=end_ind_this_task,
                        type=task_type,
                        model_version=model_version,
                    )
                )
        if task_type == elasticdl_pb2.TRAINING:
            random.shuffle(tasks)
            self._todo.extend(tasks)
        else:
            with self._lock:
                self._todo.extend(tasks)
        return tasks

    def _create_save_model_task(self):
        # For the SavedModel Task, we need a shard of data to build the model
        shards = self._training_shards
        assert shards is not None

        (shard_name, (start_ind_this_shard, num_records_this_shard)) = next(
            iter(shards.items())
        )
        start_ind_this_task = start_ind_this_shard
        end_ind_this_task = start_ind_this_shard + min(
            self._records_per_task, num_records_this_shard
        )

        # Use the first shard of data to do the SavedModel work
        save_model_task = _Task(
            shard_name=shard_name,
            start=start_ind_this_task,
            end=end_ind_this_task,
            type=elasticdl_pb2.SAVE_MODEL,
        )

        self._todo.append(save_model_task)

    def invoke_task_list_done_callback(self):
        """
        Pop a callback from the list and invoke it.
        If the callback list is empty, return False directly.
        """
        if not self._task_list_done_callbacks:
            return False

        callback = self._task_list_done_callbacks.pop()
        callback()
        return True

    def get(self, worker_id):
        """Return next (task_id, Task) tuple"""

        with self._lock:
            # TODO: check if task queue doesn't have training task,
            #       to avoid the queue is overwhelmed by evaluation tasks.
            if not self._todo and self._epoch < self._num_epochs - 1:
                # Start a new epoch
                self._epoch += 1
                self.create_tasks(elasticdl_pb2.TRAINING)
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
