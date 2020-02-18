"""TaskQueue Implementation"""

import random
import threading

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.constants import TaskExecCounterKey
from elasticdl.python.common.log_utils import default_logger as logger


class _Task(object):
    """Internal representation of a task"""

    def __init__(
        self, shard_name, start, end, type, model_version=-1, **kwargs
    ):
        self.shard_name = shard_name
        self.start = start
        self.end = end
        self.type = type
        self.model_version = model_version
        self.extended_config = kwargs

    def _info(self):
        return (
            self.shard_name,
            self.start,
            self.end,
            self.type,
            self.model_version,
        )


class JobCounter(object):
    """Counters for job"""

    def __init__(self, total_records=0, failed_records=0):
        self._total_records = total_records
        self._failed_records = failed_records

    @property
    def total_records(self):
        return self._total_records

    @total_records.setter
    def total_records(self, total_records):
        self._total_records = total_records

    @property
    def failed_records(self):
        return self._failed_records

    @failed_records.setter
    def failed_records(self, failed_records):
        self._failed_records = failed_records


class _TaskDispatcher(object):
    """Creates and dispatches Tasks. Keep track of a Task's lifecycle."""

    def __init__(
        self,
        training_shards,
        evaluation_shards,
        prediction_shards,
        records_per_task,
        num_epochs,
        callbacks_list,
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
            callbacks_list: The Keras CallbacksList object to contain all
                callback instances.
        """
        self._lock = threading.Lock()

        self._num_epochs = num_epochs
        self._epoch = 0
        self._training_shards = training_shards
        self._evaluation_shards = evaluation_shards
        self._prediction_shards = prediction_shards
        self._records_per_task = records_per_task
        self._callbacks_list = callbacks_list
        self._callbacks_list.model.stop_training = False

        self._todo = []
        # dictionary from task id to Task.
        self._doing = {}
        self._task_id = 0
        self._eval_todo = []
        self._evaluation_service = None

        # Callback list to invoke after all tasks complete.
        self._tasks_done_deferred_callbacks = []

        self._job_counters = {}

        if self._training_shards:
            logger.info("Starting epoch %d", self._epoch)
            self.create_tasks(elasticdl_pb2.TRAINING)
        elif self._evaluation_shards:
            self.create_tasks(elasticdl_pb2.EVALUATION)
        elif self._prediction_shards:
            self.create_tasks(elasticdl_pb2.PREDICTION)

    def reset_job_counters(self, task_type):
        """Return record number in specific task_type"""
        self._job_counters[task_type] = JobCounter()

    def create_tasks(self, task_type, model_version=-1):
        logger.info(
            "Creating a new set of %s tasks for model version %d",
            elasticdl_pb2._TASKTYPE.values_by_number[task_type].name.lower(),
            model_version,
        )
        self.reset_job_counters(task_type)
        if task_type == elasticdl_pb2.TRAINING:
            shards = self._training_shards
        elif task_type == elasticdl_pb2.EVALUATION:
            shards = self._evaluation_shards
        else:
            shards = self._prediction_shards
        tasks = []
        num_records_before_create = self._job_counters[task_type].total_records
        # Note that a shard may contain records for multiple tasks.
        for (
            shard_name,
            (start_ind_this_shard, num_records_this_shard),
        ) in shards.items():
            max_ind_this_shard = start_ind_this_shard + num_records_this_shard
            self._job_counters[
                task_type
            ].total_records += num_records_this_shard
            for start_ind_this_task in range(
                start_ind_this_shard,
                max_ind_this_shard,
                self._records_per_task,
            ):
                end_ind_this_task = min(
                    start_ind_this_task + self._records_per_task,
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
        elif task_type == elasticdl_pb2.EVALUATION:
            self._eval_todo.extend(tasks)
        else:
            self._todo.extend(tasks)
        logger.info(
            "%d tasks created with total of %d records."
            % (
                len(tasks),
                self._job_counters[task_type].total_records
                - num_records_before_create,
            )
        )

    def get_eval_task(self, worker_id):
        """Return next evaluation (task_id, Task) tuple"""
        with self._lock:
            if not self._eval_todo:
                return -1, None
            self._task_id += 1
            task = self._eval_todo.pop()
            self._doing[self._task_id] = (worker_id, task)
            return self._task_id, task

    def _create_train_end_callback_task(self):
        """
        Build one instance of training end task and add it to todo list.
        Because we need create a dataset to build the model for
        SavedModelExporter to execute on_train_end,we include
        a shard of data in this task.
        """

        self.reset_job_counters(elasticdl_pb2.TRAIN_END_CALLBACK)
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
        train_end_callback_task = _Task(
            shard_name=shard_name,
            start=start_ind_this_task,
            end=end_ind_this_task,
            type=elasticdl_pb2.TRAIN_END_CALLBACK,
        )

        self._todo.append(train_end_callback_task)

    def add_deferred_callback_create_train_end_task(self):
        self._tasks_done_deferred_callbacks.append(
            lambda: self._create_train_end_callback_task()
        )

    def invoke_deferred_callback(self):
        """
        Pop a callback from the list and invoke it.
        If the callback list is empty, return False directly.
        """
        if not self._tasks_done_deferred_callbacks:
            return False

        with self._lock:
            if not self._tasks_done_deferred_callbacks:
                return False

            callback = self._tasks_done_deferred_callbacks.pop()
            callback()
            return True

    def get(self, worker_id):
        """Return next (task_id, Task) tuple"""

        with self._lock:
            # TODO: check if task queue doesn't have training task,
            #       to avoid the queue is overwhelmed by evaluation tasks.
            if (
                not self._todo
                and not self._callbacks_list.model.stop_training
                and self._epoch < self._num_epochs - 1
            ):
                # Start a new epoch
                self._epoch += 1
                self.create_tasks(elasticdl_pb2.TRAINING)
                logger.info("Starting epoch %d", self._epoch)

            if self._callbacks_list.model.stop_training:
                # Clear todo list to stop training
                self._todo = []

            if not self._todo:
                # No more tasks
                return -1, None

            self._task_id += 1
            task = self._todo.pop()
            # TODO: Handle timeout of tasks.
            self._doing[self._task_id] = (worker_id, task)

            return self._task_id, task

    def report(self, request, success):
        """Report if the task is successful or not"""

        task_id = request.task_id
        evaluation_task_completed = False
        with self._lock:
            _, task = self._doing.pop(task_id, (-1, None))
            if task:
                self._job_counters[
                    task.type
                ].failed_records += request.exec_counters.get(
                    TaskExecCounterKey.FAIL_COUNT, 0
                )
            if not task:
                logger.warning("Unknown task_id: %d" % task_id)
            elif not success:
                # TODO: keep count of retries.
                if task.type == elasticdl_pb2.TRAINING:
                    self._todo.append(task)
                else:
                    self._eval_todo.append(task)
            elif (
                task.type == elasticdl_pb2.EVALUATION
                and self._evaluation_service is not None
            ):
                evaluation_task_completed = True
            else:
                self._call_on_task_end(task)
                logger.info(
                    "Task:%d completed, %d remaining tasks",
                    task_id,
                    len(self._todo) + len(self._doing),
                )
        if evaluation_task_completed:
            self._evaluation_service.complete_task()

    def finished(self):
        """Return if all tasks are done"""
        return all([not self._todo, not self._eval_todo, not self._doing])

    def recover_tasks(self, worker_id):
        """Recover doing tasks for a dead worker"""

        with self._lock:
            ids = [
                id for id, (wid, _) in self._doing.items() if wid == worker_id
            ]
        request = elasticdl_pb2.ReportTaskResultRequest()
        for id in ids:
            request.task_id = id
            self.report(request, False)

    # TODO: need to re-check after refactoring servicer.py
    def set_evaluation_service(self, evaluation_service):
        with self._lock:
            self._evaluation_service = evaluation_service
            if self._evaluation_shards and not self._training_shards:
                evaluation_service.init_eval_only_job(len(self._eval_todo))

    def _call_on_task_end(self, task):
        # The on_task_end is not a method of tf.keras.callbacks.Callback
        # and tf.keras.callbacks.CallbackList. So, we need to check
        # before calling the method.
        for callback in self._callbacks_list.callbacks:
            if hasattr(callback, "on_task_end"):
                callback.on_task_end(task)
