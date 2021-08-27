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

import threading
import time

from elasticai_api.common.constants import DefaultDatasetName
from elasticai_api.proto import elasticai_api_pb2
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.common.model_utils import (
    get_dict_from_params_str,
    get_module_file_path,
    load_module,
)
from elasticdl.python.master.dataset_shard import (
    _CHECKPOINT_VERSION,
    Dataset,
    ShardCheckpoint,
    Task,
)

_TASK_TIMEOUT_THRESHOLD_SECS = 300


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
        self._evaluation_service = None

        # Callback list to invoke after all tasks complete.
        self._tasks_done_deferred_callbacks = []

        self._load_data_reader_fn(args)
        self._datasets = {}
        if args.need_elasticdl_job_service:
            self._create_training_dataset(
                args.training_data, args.data_reader_params
            )
            self._create_evaluation_dataset(
                args.validation_data, args.data_reader_params
            )
        self._set_completed_steps_by_checkpoint(args.checkpoint_dir_for_init)
        self._add_deferred_callback_create_train_end_task()
        self._worker_start_task_time = {}
        self._task_timeout_callbacks = []

    def _load_data_reader_fn(self, args):
        if args.need_elasticdl_job_service:
            from elasticdl.python.data.reader.data_reader_factory import (
                create_data_reader,
            )

            self._create_data_reader_fn = create_data_reader
        else:
            self._create_data_reader_fn = None

        if args.model_zoo:
            # Initialize the components from the model definition
            model_module = load_module(
                get_module_file_path(args.model_zoo, args.model_def)
            ).__dict__
            if args.custom_data_reader in model_module:
                self._create_data_reader_fn = model_module[
                    args.custom_data_reader
                ]

    def _create_training_dataset(self, training_data, data_reader_params):
        shards = self._maybe_create_shards(training_data, data_reader_params)
        self._create_default_dataset(shards, DefaultDatasetName.TRAINING)

    def _create_evaluation_dataset(self, validation_data, data_reader_params):
        shards = self._maybe_create_shards(
            validation_data, data_reader_params,
        )
        self._create_default_dataset(shards, DefaultDatasetName.EVALUATION)

    def _create_default_dataset(self, shards, dataset_name):
        if shards:
            dataset = Dataset(
                shuffle_shards=self._shuffle_shards,
                records_per_task=self._records_per_task,
                dataset_size=self._dataset_size,
                num_epochs=self._num_epochs,
                dataset_name=dataset_name,
            )
            dataset.set_shards(shards)
            self._datasets[dataset_name] = dataset

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

        from elasticdl.python.common.save_utils import CheckpointSaver

        if not CheckpointSaver.check_checkpoint_valid(checkpoint_dir_for_init):
            raise ValueError(
                "Invalid checkpoint directory {}".format(
                    checkpoint_dir_for_init
                )
            )

        self._completed_steps = CheckpointSaver.get_version_from_checkpoint(
            checkpoint_dir_for_init
        )

    def set_dataset_params(
        self,
        batch_size,
        num_epochs,
        dataset_size,
        shuffle,
        shuffle_shards,
        num_minibatches_per_shard,
        dataset_name=None,
    ):
        dataset_name = (
            dataset_name if dataset_name else DefaultDatasetName.TRAINING
        )
        logger.info(
            "Set training parameters: "
            "dataset_name = {}, batch_size={}, num_epochs={}, "
            "dataset_size={}, shuffle={}, shuffle_shards={}, "
            "num_minibatches_per_shard={}".format(
                dataset_name,
                batch_size,
                num_epochs,
                dataset_size,
                shuffle,
                shuffle_shards,
                num_minibatches_per_shard,
            )
        )

        with self._lock:
            if dataset_name in self._datasets:
                logger.info(
                    "The shards for dataset {} have already been initialized."
                    "Ignore these shard parameters.".format(dataset_name)
                )
                return

            logger.info("Initialize with these training parameters.")
            # The master receives the training params to create shards
            num_minibatches_per_task = (
                num_minibatches_per_shard
                if num_minibatches_per_shard > 0
                else self._num_minibatches_per_task
            )
            records_per_task = batch_size * num_minibatches_per_task
            dataset = Dataset(
                shuffle,
                records_per_task,
                dataset_size,
                num_epochs,
                dataset_name,
            )
            if dataset_size < 0:
                logger.error(
                    "No shard creating for datset {} because dataset "
                    "size {} <= 0".format(dataset_name, dataset_size)
                )
            self._datasets[dataset_name] = dataset

    def create_evaluation_tasks(self, model_version):
        """ Create evaluation tasks and return the number of
        evaluation tasks.
        """
        eval_dataset = self._datasets[DefaultDatasetName.EVALUATION]
        eval_dataset.create_tasks(model_version)
        return len(eval_dataset.todo)

    def _create_train_end_callback_task(self):
        """
        Build one instance of training end task and add it to todo list.
        Because we need create a dataset to build the model for
        SavedModelExporter to execute on_train_end,we include
        a shard of data in this task.
        """
        training_dataset = self._datasets.get(
            DefaultDatasetName.TRAINING, None
        )
        if not training_dataset:
            return

        shards = training_dataset.get_shards()
        assert shards is not None

        (shard_name, start_idx_this_shard, num_records_this_shard) = next(
            iter(shards)
        )
        start_idx_this_task = start_idx_this_shard
        end_idx_this_task = start_idx_this_shard + min(
            self._records_per_task, num_records_this_shard
        )

        # Use the first shard of data to do the SavedModel work
        train_end_callback_task = Task(
            shard_name=shard_name,
            start=start_idx_this_task,
            end=end_idx_this_task,
            type=elasticai_api_pb2.TRAIN_END_CALLBACK,
        )
        training_dataset.todo.append(train_end_callback_task)

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

    def get_dataset_task(self, worker_id, dataset_name):
        """Return next (task_id, Task) tuple"""
        with self._lock:
            dataset = self.get_dataset(dataset_name)
            if dataset:
                task_id, task = dataset.get_task(
                    worker_id, self.support_fault_tolerance
                )
                return task_id, task
            else:
                return -1, None

    def reset_dataset(self, dataset_name):
        with self._lock:
            dataset = self.get_dataset(dataset_name)
            if dataset:
                dataset.reset()
                logger.info("Dataset {} reset".format(dataset_name))

    def get_dataset(self, dataset_name):
        return self._datasets.get(dataset_name, None)

    def report_dataset_task(self, request, success):
        """Report if the task is successful or not"""

        task_id = request.task_id
        dataset_name = request.dataset_name
        with self._lock:
            return self._report_task(dataset_name, task_id, success)

    def _report_task(self, dataset_name, task_id, success):
        dataset = self._datasets.get(dataset_name, None)
        if not dataset:
            raise ValueError(
                "There is no dataset shard for the dataset {}".format(
                    dataset_name
                )
            )
        worker_id, task, start_time = dataset.doing.pop(
            task_id, (-1, None, -1)
        )
        evaluation_task_completed = False

        if not task:
            logger.warning(
                "Unknown task_id: %d of dataset %s" % (task_id, dataset_name)
            )
        elif not success:
            logger.warning("Task %d of %s failed " % (task_id, dataset_name))
            dataset.recover_task(task)
        else:
            if (
                dataset_name == DefaultDatasetName.EVALUATION
                and self._evaluation_service is not None
            ):
                # Evaluation of ElasticDL training loop
                evaluation_task_completed = True
            if dataset_name == DefaultDatasetName.TRAINING:
                # Max step of ElasticDL training loop
                self._check_exceed_max_step(task)
            logger.info(
                "Task:%d completed, %d remaining tasks for Dataset %s",
                task_id,
                len(dataset.todo) + len(dataset.doing),
                dataset_name,
            )

        if evaluation_task_completed:
            self._evaluation_service.complete_task()

        return (time.time() - start_time), task, worker_id

    def _check_exceed_max_step(self, task):
        if self._max_step > 0:
            task_records = task.shard.end - task.shard.start
            task_batch_count = int(task_records / self._batch_size)
            self._completed_steps += task_batch_count
            if self._completed_steps > self._max_step:
                self._datasets[DefaultDatasetName.TRAINING].todo.clear()
                self._should_stop = True

    def finished(self):
        """Return if all tasks are done"""
        finished = all([ds.completed() for _, ds in self._datasets.items()])
        return finished

    def recover_tasks(self, worker_id):
        """Recover doing tasks for a dead worker if needed"""
        if not self.support_fault_tolerance:
            return

        logger.info("Recover the tasks assigned to worker %d" % worker_id)

        for name, dataset in self._datasets.items():
            ids = [
                id
                for id, (wid, _, _) in dataset.doing.items()
                if wid == worker_id
            ]
            request = elasticai_api_pb2.ReportDatasetTaskResultRequest()
            for id in ids:
                request.task_id = id
                request.dataset_name = name
                self.report_dataset_task(request, False)

    # TODO: need to re-check after refactoring servicer.py
    def set_evaluation_service(self, evaluation_service):
        with self._lock:
            self._evaluation_service = evaluation_service
            training_dataset = self._datasets.get(
                DefaultDatasetName.TRAINING, None
            )
            eval_dataset = self._datasets.get(
                DefaultDatasetName.EVALUATION, None
            )
            if eval_dataset and not training_dataset:
                evaluation_service.init_eval_only_job(len(eval_dataset.todo))

    def start(self):
        if self.support_fault_tolerance and self.relaunch_timeout_worker:
            threading.Thread(
                target=self._check_and_reassign_timeout_tasks,
                name="check_timeout_tasks",
                daemon=True,
            ).start()

    def reset_worker_start_task_time(self, worker_id):
        self._worker_start_task_time[worker_id] = time.time()

    def record_task_completed_time(self, dataset_name, completed_time):
        dataset = self._datasets[dataset_name]
        dataset.max_task_completed_time = max(
            dataset.max_task_completed_time, completed_time
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
            for _, dataset in self._datasets.items():
                doing_tasks = dataset.doing.copy()
                cur_time = time.time()
                for _, (worker_id, task, _) in doing_tasks.items():
                    start_time = self._worker_start_task_time[worker_id]
                    if cur_time - start_time > max(
                        _TASK_TIMEOUT_THRESHOLD_SECS,
                        3 * dataset.max_task_completed_time,
                    ):
                        logger.info(
                            "worker %d timeout, relaunch it" % worker_id
                        )
                        self.recover_tasks(worker_id)
                        # TODO: save worker logs before remove it
                        self._invoke_task_timeout_callback(worker_id)
                        break
            time.sleep(30)

    def get_shard_checkpoint(self, dataset_name):
        """Get the data shard checkpoint by dataset name.

        Args:
            dataset_name: string

        Returns:
            ShardCheckpoint.
        """
        dataset_name = (
            dataset_name if dataset_name else DefaultDatasetName.TRAINING
        )
        with self._lock:
            if dataset_name in self._datasets:
                dataset = self._datasets[dataset_name]
                return dataset.get_checkpoint()
            else:
                return None

    def restore_shard_from_checkpoint(self, checkpoint):
        try:
            checkpoint = ShardCheckpoint.from_json(checkpoint)
            if checkpoint.version != _CHECKPOINT_VERSION:
                return False
            checkpoint.dataset_name = (
                checkpoint.dataset_name
                if checkpoint.dataset_name
                else DefaultDatasetName.TRAINING
            )

            dataset = Dataset.restore_from_shard_checkpoint(checkpoint)
            with self._lock:
                self._datasets[checkpoint.dataset_name] = dataset
            logger.info(
                "Restore {} dataset shards from checkpoint".format(
                    checkpoint.dataset_name
                )
            )
            return True
        except Exception as e:
            logger.error("Fail to restore shards from the checkpoint", e)

        return False

    def get_dataset_epoch(self, dataset_name):
        if dataset_name in self._datasets:
            return self._datasets[dataset_name].get_epoch()
        else:
            logger.error("There is not exit dataset {}".format(dataset_name))
            return 0
