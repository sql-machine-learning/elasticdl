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

"""TaskQueue Implementation"""

import random
import threading
import time

from elasticai_api.common.constants import TaskExecCounterKey
from elasticai_api.proto import elasticai_api_pb2
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.common.model_utils import (
    get_dict_from_params_str,
    get_module_file_path,
    load_module,
)
from elasticdl.python.common.save_utils import CheckpointSaver
from elasticdl.python.data.reader.data_reader_factory import create_data_reader

_MAX_TASK_RETRIES = 3
_TASK_TIMEOUT_THRESHOLD_SECS = 300


class _Shard(object):
    def __init__(self, name, start, end, indices=None):
        self.name = name
        self.start = start
        self.end = end
        self.indices = indices


class _Task(object):
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
        self.shard = _Shard(shard_name, start, end, task_record_indices)
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


class TaskManager(object):
    """Creates and dispatches Tasks. Keep track of a Task's lifecycle."""

    def __init__(
        self, args,
    ):
        """
        Args: The return of the argparse and it must contain the
            following arguments:
            training_data: Training data definition like a recordio
                directory or a MaxCompute table
            validation_data: Validation data definition like a recordio
                directory or a MaxCompute table
            minibatch_size: The iteration batch size of each worker.
            num_minibatches_per_task: The batch count per task.
            num_epochs: The total number of epochs for the tasks where
                an epoch is a complete iteration over the shards.
            max_step: The maximum iteration step.
            data_reader_params: Parameters to create a data reader.
                For example, the column name used to create ODPSReader.
            model_zoo: The folder name of model zoo
            model_def: The absolute path of the model definition file.
            custom_data_reader: The function name of custom data reader.
        """
        self._lock = threading.Lock()

        self._batch_size = args.minibatch_size
        self._num_epochs = args.num_epochs
        self._dataset_size = None
        self._shuffle = False
        self._shuffle_shards = False
        self.support_fault_tolerance = args.task_fault_tolerance
        self.relaunch_timeout_worker = args.relaunch_timeout_worker
        self._epoch = 0
        self._max_step = args.max_step
        self._completed_steps = 0
        self._num_minibatches_per_task = args.num_minibatches_per_task
        self._records_per_task = (
            args.minibatch_size * args.num_minibatches_per_task
        )

        self._should_stop = False

        self._todo = []
        # dictionary from task id to Task.
        self._doing = {}
        self._task_id = 0
        self._eval_todo = []
        self._evaluation_service = None

        # Callback list to invoke after all tasks complete.
        self._tasks_done_deferred_callbacks = []

        self._job_counters = {}
        self._task_retry_count = {}
        self._load_data_reader_fn(args)
        self._create_training_tasks(
            args.training_data, args.data_reader_params
        )
        self._create_evaluation_tasks(
            args.validation_data, args.data_reader_params
        )
        self._set_completed_steps_by_checkpoint(args.checkpoint_dir_for_init)
        if not args.custom_training_loop:
            self._add_deferred_callback_create_train_end_task()

        self._max_task_completed_times = {
            elasticai_api_pb2.EVALUATION: 0,
            elasticai_api_pb2.TRAINING: 0,
        }
        self._worker_start_task_time = {}
        self._task_timeout_callbacks = []

    def _load_data_reader_fn(self, args):
        self._create_data_reader_fn = create_data_reader

        if args.model_zoo:
            # Initialize the components from the model definition
            model_module = load_module(
                get_module_file_path(args.model_zoo, args.model_def)
            ).__dict__
            if args.custom_data_reader in model_module:
                self._create_data_reader_fn = model_module[
                    args.custom_data_reader
                ]

    def _create_training_tasks(self, training_data, data_reader_params):
        self._training_shards = self._maybe_create_shards(
            training_data, data_reader_params
        )
        if self._training_shards:
            logger.info("Starting epoch %d", self._epoch)
            self.create_tasks(elasticai_api_pb2.TRAINING)

    def _create_evaluation_tasks(self, validation_data, data_reader_params):
        self._evaluation_shards = self._maybe_create_shards(
            validation_data, data_reader_params,
        )
        if not self._training_shards and self._evaluation_shards:
            self.create_tasks(elasticai_api_pb2.EVALUATION)

    def _maybe_create_shards(self, data_origin, data_reader_params):
        kwargs = get_dict_from_params_str(data_reader_params)
        partition = kwargs.get("partition", None) if kwargs else None
        return (
            self._create_data_reader_fn(
                data_origin=data_origin,
                records_per_task=self._records_per_task,
                partition=partition,
            ).create_shards()
            if data_origin
            else {}
        )

    def _set_completed_steps_by_checkpoint(self, checkpoint_dir_for_init):
        if not checkpoint_dir_for_init:
            return

        if not CheckpointSaver.check_checkpoint_valid(checkpoint_dir_for_init):
            raise ValueError(
                "Invalid checkpoint directory {}".format(
                    checkpoint_dir_for_init
                )
            )

        self._completed_steps = CheckpointSaver.get_version_from_checkpoint(
            checkpoint_dir_for_init
        )

    def set_training_params(
        self, batch_size, num_epochs, dataset_size, shuffle, shuffle_shards
    ):
        logger.info(
            "Set training parameters: "
            "batch_size={}, num_epochs={}, dataset_size={},"
            "shuffle={}, shuffle_shards={}".format(
                batch_size, num_epochs, dataset_size, shuffle, shuffle_shards
            )
        )

        with self._lock:
            if not self._training_shards:
                # The master receives the training params to create shards
                self._batch_size = batch_size
                self._shuffle = shuffle
                self._shuffle_shards = shuffle_shards
                self._records_per_task = (
                    batch_size * self._num_minibatches_per_task
                )
                self._num_epochs = (
                    num_epochs if num_epochs > 0 else self._num_epochs
                )
                self._dataset_size = (
                    dataset_size if dataset_size > 0 else self._dataset_size
                )
                self._training_shards = self._create_shards_by_dataset_size(
                    dataset_size
                )
                if self._training_shards:
                    logger.info("Starting epoch %d", self._epoch)
                    self.create_tasks(elasticai_api_pb2.TRAINING)

    def _create_shards_by_dataset_size(self, dataset_size):
        shards = []
        num_shards = dataset_size // self._records_per_task
        start_idx = 0
        for shard_id in range(num_shards):
            shards.append(("", start_idx, self._records_per_task,))
            start_idx += self._records_per_task
        # Create a shard with the last records
        num_records_left = dataset_size % self._records_per_task
        if num_records_left != 0:
            shards.append(("", start_idx, num_records_left,))
        logger.info("Create {} shards".format(len(shards)))
        return shards

    def reset_job_counters(self, task_type):
        """Return record number in specific task_type"""
        self._job_counters[task_type] = JobCounter()

    def create_tasks(self, task_type, model_version=-1):
        logger.info(
            "Creating a new set of %s tasks for model version %d",
            elasticai_api_pb2._TASKTYPE.values_by_number[
                task_type
            ].name.lower(),
            model_version,
        )
        self.reset_job_counters(task_type)
        if task_type == elasticai_api_pb2.TRAINING:
            shards = self._training_shards
        elif task_type == elasticai_api_pb2.EVALUATION:
            shards = self._evaluation_shards
        else:
            raise ValueError("Not supported type")
        tasks = []
        num_records_before_create = self._job_counters[task_type].total_records
        # Note that a shard may contain records for multiple tasks.
        if self._shuffle:
            record_indices = list(range(0, self._dataset_size))
            random.shuffle(record_indices)
        for (
            shard_name,
            start_idx_this_shard,
            num_records_this_shard,
        ) in shards:
            max_idx_this_shard = start_idx_this_shard + num_records_this_shard
            self._job_counters[
                task_type
            ].total_records += num_records_this_shard
            for start_idx_this_task in range(
                start_idx_this_shard,
                max_idx_this_shard,
                self._records_per_task,
            ):
                end_idx_this_task = min(
                    start_idx_this_task + self._records_per_task,
                    max_idx_this_shard,
                )
                task_record_indices = (
                    record_indices[start_idx_this_task:end_idx_this_task]
                    if self._shuffle
                    else None
                )

                # Note that only records in [start, end) of this task
                # will be consumed later in the worker that handles
                # this task.
                tasks.append(
                    _Task(
                        shard_name=shard_name,
                        start=start_idx_this_task,
                        end=end_idx_this_task,
                        type=task_type,
                        model_version=model_version,
                        task_record_indices=task_record_indices,
                    )
                )
        if task_type == elasticai_api_pb2.TRAINING:
            if self._shuffle_shards:
                random.shuffle(tasks)
            self._todo.extend(tasks)
        elif task_type == elasticai_api_pb2.EVALUATION:
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

    def create_evaluation_tasks(self, model_version):
        """ Create evaluation tasks and return the number of
        evaluation tasks.
        """
        self.create_tasks(elasticai_api_pb2.EVALUATION, model_version)
        return len(self._eval_todo)

    def get_eval_task(self, worker_id):
        """Return next evaluation (task_id, Task) tuple"""
        with self._lock:
            if not self._eval_todo:
                return -1, None
            self._task_id += 1
            task = self._eval_todo.pop(0)
            if self.support_fault_tolerance:
                self._doing[self._task_id] = (worker_id, task, time.time())
            return self._task_id, task

    def _create_train_end_callback_task(self):
        """
        Build one instance of training end task and add it to todo list.
        Because we need create a dataset to build the model for
        SavedModelExporter to execute on_train_end,we include
        a shard of data in this task.
        """
        if not self._training_shards:
            return

        self.reset_job_counters(elasticai_api_pb2.TRAIN_END_CALLBACK)
        shards = self._training_shards
        assert shards is not None

        (shard_name, start_idx_this_shard, num_records_this_shard) = next(
            iter(shards)
        )
        start_idx_this_task = start_idx_this_shard
        end_idx_this_task = start_idx_this_shard + min(
            self._records_per_task, num_records_this_shard
        )

        # Use the first shard of data to do the SavedModel work
        train_end_callback_task = _Task(
            shard_name=shard_name,
            start=start_idx_this_task,
            end=end_idx_this_task,
            type=elasticai_api_pb2.TRAIN_END_CALLBACK,
        )

        self._todo.append(train_end_callback_task)

    def _add_deferred_callback_create_train_end_task(self):
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
            if (
                not self._todo
                and not self._should_stop
                and self._epoch < self._num_epochs - 1
            ):
                # Start a new epoch
                self._epoch += 1
                logger.info("Starting epoch %d", self._epoch)
                self.create_tasks(elasticai_api_pb2.TRAINING)

            if not self._todo:
                # No more tasks
                return -1, None

            self._task_id += 1
            task = self._todo.pop(0)
            if self.support_fault_tolerance:
                self._doing[self._task_id] = (worker_id, task, time.time())

            return self._task_id, task

    def report(self, request, success):
        """Report if the task is successful or not"""

        task_id = request.task_id
        evaluation_task_completed = False
        with self._lock:
            worker_id, task, start_time = self._doing.pop(
                task_id, (-1, None, -1)
            )
            if task:
                self._job_counters[
                    task.type
                ].failed_records += request.exec_counters.get(
                    TaskExecCounterKey.FAIL_COUNT, 0
                )
            if not task:
                logger.warning("Unknown task_id: %d" % task_id)
            elif not success:
                logger.warning("Task %d of %s failed " % (task_id, task.type))
                if not self.check_exceed_max_task_retries(task):
                    if task.type in [
                        elasticai_api_pb2.TRAINING,
                        elasticai_api_pb2.TRAIN_END_CALLBACK,
                    ]:
                        self._todo.append(task)
                    else:
                        self._eval_todo.append(task)
            elif (
                task.type == elasticai_api_pb2.EVALUATION
                and self._evaluation_service is not None
            ):
                evaluation_task_completed = True
            else:
                self._check_exceed_max_step(task)
                logger.info(
                    "Task:%d completed, %d remaining tasks",
                    task_id,
                    len(self._todo) + len(self._doing),
                )
            if evaluation_task_completed:
                self._evaluation_service.complete_task()

            if success:
                if task in self._task_retry_count:
                    del self._task_retry_count[task]

        return (time.time() - start_time), task, worker_id

    def _check_exceed_max_step(self, task):
        if self._max_step > 0 and task.type == elasticai_api_pb2.TRAINING:
            task_records = task.shard.end - task.shard.start
            task_batch_count = int(task_records / self._batch_size)
            self._completed_steps += task_batch_count
            if self._completed_steps > self._max_step:
                self._todo.clear()
                self._should_stop = True

    def check_exceed_max_task_retries(self, task):
        self._task_retry_count.setdefault(task, 1)
        self._task_retry_count[task] += 1
        if self._task_retry_count[task] > _MAX_TASK_RETRIES:
            logger.error(
                "A %s task failed with %d retries "
                % (task.type, _MAX_TASK_RETRIES)
            )
            return True
        return False

    def finished(self):
        """Return if all tasks are done"""
        return all([not self._todo, not self._doing])

    def recover_tasks(self, worker_id):
        """Recover doing tasks for a dead worker if needed"""
        if not self.support_fault_tolerance:
            return

        logger.info("Recover the tasks assigned to worker %d" % worker_id)

        with self._lock:
            ids = [
                id
                for id, (wid, _, _) in self._doing.items()
                if wid == worker_id
            ]
        request = elasticai_api_pb2.ReportTaskResultRequest()
        for id in ids:
            request.task_id = id
            self.report(request, False)

    # TODO: need to re-check after refactoring servicer.py
    def set_evaluation_service(self, evaluation_service):
        with self._lock:
            self._evaluation_service = evaluation_service
            if self._evaluation_shards and not self._training_shards:
                evaluation_service.init_eval_only_job(len(self._eval_todo))

    def start(self):
        if self.support_fault_tolerance and self.relaunch_timeout_worker:
            threading.Thread(
                target=self._check_and_reassign_timeout_tasks,
                name="check_timeout_tasks",
                daemon=True,
            ).start()

    def reset_worker_start_task_time(self, worker_id):
        self._worker_start_task_time[worker_id] = time.time()

    def record_task_completed_time(self, task_type, completed_time):
        self._max_task_completed_times[task_type] = max(
            self._max_task_completed_times[task_type], completed_time
        )

    def set_task_timeout_callback(self, callback_fn):
        self._task_timeout_callbacks.append(callback_fn)

    def _invoke_task_timeout_callback(self, worker_id):
        for callback_fn in self._task_timeout_callbacks:
            callback_fn(worker_id)

    def _check_and_reassign_timeout_tasks(self):
        """Check whether there are timeout tasks periodically.
        """
        while True:
            doing_tasks = self._doing.copy()
            cur_time = time.time()
            for _, (worker_id, task, start_time) in doing_tasks.items():
                if task.type == elasticai_api_pb2.TRAINING:
                    start_time = self._worker_start_task_time[worker_id]
                if task.type in [
                    elasticai_api_pb2.TRAINING,
                    elasticai_api_pb2.EVALUATION,
                ]:
                    if cur_time - start_time > max(
                        _TASK_TIMEOUT_THRESHOLD_SECS,
                        3 * self._max_task_completed_times[task.type],
                    ):
                        logger.info(
                            "worker %d timeout, relaunch it" % worker_id
                        )
                        self.recover_tasks(worker_id)
                        # TODO: save worker logs before remove it
                        self._invoke_task_timeout_callback(worker_id)
                        break
            time.sleep(30)
