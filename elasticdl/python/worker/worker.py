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
import traceback
from distutils.version import LooseVersion

import tensorflow as tf

from elasticai_api.common.data_shard_service import DataShardService
from elasticai_api.proto import elasticai_api_pb2
from elasticdl.python.common.constants import JobType, MetricsDictKey, Mode
from elasticdl.python.common.log_utils import get_logger
from elasticdl.python.common.model_handler import ModelHandler
from elasticdl.python.common.model_utils import (
    get_dict_from_params_str,
    get_model_spec,
    get_training_func_spec,
    set_callback_parameters,
)
from elasticdl.python.common.timing_utils import Timing
from elasticdl.python.elasticdl.callbacks import SavedModelExporter
from elasticdl.python.worker.allreduce_trainer import AllReduceTrainer
from elasticdl.python.worker.ps_trainer import ParameterServerTrainer
from elasticdl.python.worker.task_data_service import TaskDataService
from elasticdl_client.common.constants import DistributionStrategy

# The default maximum number of a minibatch retry as its results
# (e.g. gradients) are not accepted by master.
DEFAULT_MAX_MINIBATCH_RETRY_NUM = 64

DEFAULT_STEPS_TO_CHECK_RENDEZVOUS = 20

_IS_TF2 = LooseVersion(tf.__version__) >= LooseVersion("2.0.0")


class Worker(object):
    """ElasticDL worker"""

    def __init__(
        self,
        args,
        master_client=None,
        ps_client=None,
        max_minibatch_retry_num=DEFAULT_MAX_MINIBATCH_RETRY_NUM,
        set_parallelism=False,
    ):
        """
        Arguments:
            channel: The channel for the gRPC master service.
            ps_channels: The PS channels for PS service
            max_minibatch_retry_num: The maximum number of a minibatch retry
                as its results (e.g. gradients) are not accepted by master.
            max_allreduce_retry_num: The maximum number of retries for
                allreduce operation if allreduce-based distributed
                training strategy is used.
        """
        self._args = args
        self.logger = get_logger("Worker", level=args.log_level.upper())

        if set_parallelism:
            # Explicitly setting the parallelism will avoid multi-process hangs
            # Maybe due to an unknown bug in Tensorflow?
            # Must called before TensorFlow is initialized.
            # Not set_parallelism by default to make unittests happy.
            num_threads = os.cpu_count()
            tf.config.threading.set_inter_op_parallelism_threads(num_threads)
            tf.config.threading.set_intra_op_parallelism_threads(num_threads)

        self._mc = master_client
        self._ps_client = ps_client
        self._distribution_strategy = args.distribution_strategy
        self._max_minibatch_retry_num = max_minibatch_retry_num
        self._timing = Timing(args.log_level.upper() == "DEBUG", self.logger)
        self._log_loss_count = 0
        self._var_created = False
        self._custom_training_loop = args.custom_training_loop
        self._job_type = args.job_type
        self._minibatch_size = args.minibatch_size
        self._data_shard_service = DataShardService(
            self._mc, self._minibatch_size
        )
        if self._custom_training_loop:
            self._init_training_func_from_args(args)
        else:
            self._init_model_from_args(args)
        self._init_task_data_service(args)
        self._init_default_feed_if_needed()
        if not self._custom_training_loop:
            self._init_callbacks(args)
            self._init_trainer(args)

    def _init_model_from_args(self, args):
        """
        Please refer to elastic/python/common/args.py for more
        details about arguments of a worker.
        """
        self._log_loss_steps = args.log_loss_steps
        (
            model_inst,
            self._feed,
            loss,
            opt_fn,
            self._eval_metrics_fn,
            self._prediction_outputs_processor,
            self._custom_data_reader,
            self._callbacks_list,
        ) = get_model_spec(
            model_zoo=args.model_zoo,
            model_def=args.model_def,
            feed=args.feed,
            loss=args.loss,
            optimizer=args.optimizer,
            eval_metrics_fn=args.eval_metrics_fn,
            prediction_outputs_processor=args.prediction_outputs_processor,
            custom_data_reader=args.custom_data_reader,
            callbacks=args.callbacks,
        )

        self._model_handler = ModelHandler.get_model_handler(
            self._distribution_strategy, checkpoint_dir=args.checkpoint_dir
        )
        self._model_inst = self._model_handler.get_model_to_train(model_inst)
        self._model_inst.optimizer = opt_fn()
        self._model_inst.loss = loss
        self._model_version = -1
        self._get_model_steps = args.get_model_steps

    def _init_task_data_service(self, args):
        self._task_data_service = TaskDataService(
            self._data_shard_service,
            custom_data_reader=self._custom_data_reader,
            data_reader_params=get_dict_from_params_str(
                args.data_reader_params
            ),
            data_origin=args.training_data,
        )

    def _init_callbacks(self, args):
        saved_model_exporter = SavedModelExporter(
            self._task_data_service, self._feed, self._model_handler
        )
        # Place default callbacks at the head to execute them firstly
        self._callbacks_list.callbacks.insert(0, saved_model_exporter)
        self._callbacks_list.set_model(self._model_inst)
        set_callback_parameters(
            self._callbacks_list,
            batch_size=args.minibatch_size,
            saved_model_path=args.output,
            checkpoint_path=args.checkpoint_dir,
        )
        self._saved_model_path = args.output

    def _init_trainer(self, args):
        if self._distribution_strategy == DistributionStrategy.ALLREDUCE:
            self._trainer = AllReduceTrainer(self._mc, self._model_inst)
        elif (
            self._distribution_strategy
            == DistributionStrategy.PARAMETER_SERVER
        ):
            self._trainer = ParameterServerTrainer(
                self._model_inst, self._ps_client, self._timing, args
            )

    def _init_training_func_from_args(self, args):
        self._job_type = args.job_type
        (
            self._training_func,
            self._feed,
            self._custom_data_reader,
        ) = get_training_func_spec(
            model_zoo=args.model_zoo,
            model_def=args.model_def,
            feed=args.feed,
            custom_data_reader=args.custom_data_reader,
        )

    def _init_default_feed_if_needed(self):
        if self._feed is None:
            if hasattr(self._task_data_service.data_reader, "default_feed"):
                self._feed = (
                    self._task_data_service.data_reader.default_dataset_fn()
                )
            else:
                raise ValueError(
                    "feed is required if the data_reader used does "
                    "not provide default implementation of feed"
                )

    def _process_minibatch(
        self,
        task_type,
        features,
        labels,
        min_model_version,
        train_with_local_model=False,
    ):
        self._trainer.init_variables_if_need(features, labels)
        self._timing.start_record_time("batch_process")
        for _ in range(self._max_minibatch_retry_num):
            if task_type == elasticai_api_pb2.EVALUATION:
                self._trainer.evaluate_minibatch(features, labels)
                break
            elif task_type == elasticai_api_pb2.TRAINING:
                # TODO: optimize the logic to avoid unnecessary
                #       get_model call.
                self._callbacks_list.on_train_batch_begin(self._model_version)
                (
                    *accepted,
                    min_model_version,
                    loss,
                ) = self._trainer.train_minibatch(
                    features, labels, train_with_local_model
                )
                self._model_version = self._trainer.get_model_version()

                if accepted:
                    if (
                        self._model_version
                        >= self._log_loss_count * self._log_loss_steps
                    ):
                        self.logger.info(
                            "Loss = {}, steps = {}".format(
                                loss.numpy(), self._model_version
                            )
                        )
                        self._log_loss_count = (
                            int(self._model_version / self._log_loss_steps) + 1
                        )
                    break
            elif task_type == elasticai_api_pb2.PREDICTION:
                accepted = self._trainer.predict_minibatch(features)
                if accepted:
                    break
            else:
                raise RuntimeError("Unrecognized task type, %s" % task_type)
        else:
            # Worker got stuck, fail the task.
            # TODO: stop the worker if it fails to make any
            #       progress for some time.
            raise RuntimeError("Worker got stuck")
        self._timing.end_record_time("batch_process")
        return min_model_version

    def _process_evaluation_if_exist(self, dataset):
        """
        Check if there are evaluation tasks and process the tasks if any.
        Return:
            A python bool indicating whether worker processed some evaluation
            tasks.
        """
        evaluation_exist = False
        err_msg = ""
        for dataset_batch in dataset:
            evaluation_exist = True
            data_err_msg = self._safe_process_minibatch(
                dataset_batch, elasticai_api_pb2.EVALUATION, None
            )
            if data_err_msg:
                err_msg = data_err_msg
                break
        if evaluation_exist:
            evaluation_result = self._trainer.get_evaluation_result()
            self._mc.report_evaluation_metrics(
                model_outputs=evaluation_result[MetricsDictKey.MODEL_OUTPUT],
                labels=evaluation_result[MetricsDictKey.LABEL],
            )
            task_id = self._task_data_service.current_eval_task.task_id
            self._mc.report_task_result(task_id, err_msg)
            self._trainer.reset_evaluation_result()
        return evaluation_exist

    def _process_train_end_callback_task_if_needed(self):
        train_end_task = self._task_data_service.get_train_end_callback_task()
        if train_end_task:
            self._callbacks_list.on_train_end()
            self._mc.report_task_result(
                task_id=train_end_task.task_id, err_msg=""
            )
        if self._distribution_strategy == DistributionStrategy.ALLREDUCE:
            self._trainer.export_saved_model(self._saved_model_path)

    def _safe_process_minibatch(
        self,
        dataset_batch,
        task_type,
        model_version,
        train_with_local_model=False,
    ):
        err_msg = ""
        try:
            if self._job_type == JobType.PREDICTION_ONLY:
                features = dataset_batch
                labels = None
            else:
                features = dataset_batch[0]
                labels = dataset_batch[1]
            self._process_minibatch(
                task_type,
                features,
                labels,
                model_version,
                train_with_local_model,
            )
        except RuntimeError as err:
            err_msg = str(err)
            traceback.print_exc()
        except Exception as ex:
            err_msg = str(ex)
            traceback.print_exc()
            raise ex
        return err_msg

    def _train_and_evaluate(self):
        """
        Train and evaluate the model on the worker
        """

        # The worker needs to get model from PS if
        # `train_with_local_model=False`. This happens when:
        #     processing first minibatch
        #     any evaluation task has been executed just before this minibatch
        #     last minibatch is training task and failed
        #     local_update_count >= worker._get_model_steps
        # Otherwise, worker trains with local model, i.e.
        # `train_with_local_model=True`
        train_with_local_model = False

        # Initialize `local_update_count=get_model_steps` in order to set
        # `train_with_local_model` to False inside for-loop for the first
        # minibatch.

        local_update_count = self._get_model_steps
        last_training_minibatch_failed = False
        evaluation_task_executed = False
        dataset = self._task_data_service.get_dataset()

        dataset = self._feed(
            dataset,
            Mode.TRAINING,
            self._task_data_service.data_reader.metadata,
        )
        dataset = dataset.batch(self._minibatch_size).prefetch(1)
        self._timing.start_record_time("task_process")
        if isinstance(self._trainer, AllReduceTrainer):
            self._trainer.notify_training_loop_start()
        for dataset_batch in dataset:
            if self._job_type == JobType.TRAINING_WITH_EVALUATION:
                # Give the worker a chance to process an evaluation task
                # during training if the task exists
                evaluation_task_executed = (
                    True if self._evaluate_only() else evaluation_task_executed
                )

            task = self._task_data_service.get_current_task()
            if (
                evaluation_task_executed
                or last_training_minibatch_failed
                or local_update_count >= self._get_model_steps
            ):
                local_update_count = 0
                train_with_local_model = False
            else:
                train_with_local_model = True

            err_msg = self._safe_process_minibatch(
                dataset_batch,
                task.type,
                task.model_version,
                train_with_local_model,
            )

            local_update_count += 1
            if err_msg:
                last_training_minibatch_failed = True
            else:
                last_training_minibatch_failed = False
                if local_update_count < self._get_model_steps:
                    self._update_local_model()
            if self._task_data_service.report_record_done(
                self._minibatch_size, err_msg
            ):
                self._timing.end_record_time("task_process")
                self._timing.report_timing(reset=True)
                self._timing.start_record_time("task_process")

        if isinstance(self._trainer, AllReduceTrainer):
            self._trainer.notify_training_loop_end()

        del dataset
        # New evaluation tasks may be created after this worker's
        # training tasks are done, as other workers' may still
        # have pending training tasks.
        if self._job_type == JobType.TRAINING_WITH_EVALUATION:
            evaluation_task_executed = self._evaluate_only()

        self._process_train_end_callback_task_if_needed()

    def _evaluate_only(self):
        """
        Only evaluate the model on the worker.
        """
        evaluation_task_executed = False
        with tf.device("/device:cpu:0"):
            dataset = self._task_data_service.get_eval_dataset()

        dataset = self._feed(
            dataset,
            Mode.EVALUATION,
            self._task_data_service.data_reader.metadata,
        )
        dataset = dataset.batch(self._minibatch_size).prefetch(1)
        while True:
            # The dataset will re-call generator each time when
            # calling iterator of the dataset.
            evaluation_exist = self._process_evaluation_if_exist(dataset)
            if not evaluation_exist:
                break
        del dataset
        evaluation_task_executed = True
        return evaluation_task_executed

    def _predict_only(self):
        """
        Only predict outputs of the model with data in tasks on the worker.
        """
        while True:
            dataset = self._task_data_service.get_dataset()
            if not dataset:
                break
            dataset = self._feed(
                dataset,
                Mode.PREDICTION,
                self._task_data_service.data_reader.metadata,
            )
            dataset = dataset.batch(self._minibatch_size).prefetch(1)
            for dataset_batch in dataset:
                task = self._task_data_service.get_current_task()

                err_msg = self._safe_process_minibatch(
                    dataset_batch, task.type, task.model_version
                )
                self._task_data_service.report_record_done(
                    self._minibatch_size, err_msg
                )
            del dataset

    def run(self):
        """
        Fetches task from master with and performs training, evaluation
        or prediction.
        """
        if self._job_type == JobType.PREDICTION_ONLY:
            self._predict_only()
        elif self._job_type == JobType.EVALUATION_ONLY:
            self._evaluate_only()
        else:
            if self._custom_training_loop:
                self._elastic_allreduce_train()
            else:
                self._train_and_evaluate()

    def _elastic_allreduce_train(self):
        """
        Train and evaluate the model on the worker
        """
        if os.getenv("USE_TORCH", None):
            from elasticai_api.pytorch.controller import (
                PyTorchAllReduceController,
            )

            elastic_controller = PyTorchAllReduceController(
                self._mc, self._data_shard_service
            )
        elif _IS_TF2:
            from elasticai_api.tensorflow.controller import (
                TensorFlowV2AllReduceController,
            )

            elastic_controller = TensorFlowV2AllReduceController(
                self._mc, self._data_shard_service
            )
        else:
            from elasticai_api.tensorflow.controller import (
                TensorFlowV1AllReduceController,
            )

            elastic_controller = TensorFlowV1AllReduceController(
                self._mc, self._master_addr
            )
        # Initialize Horovod locally to generate varibles of the model
        # and optimizer.
        elastic_controller.init_horovod_locally()
        dataset = self._task_data_service.get_dataset()
        dataset = self._feed(
            dataset,
            Mode.TRAINING,
            self._task_data_service.data_reader.metadata,
        )
        dataset = dataset.batch(self._minibatch_size).prefetch(1)
        self._training_func(dataset, elastic_controller)
        del dataset
