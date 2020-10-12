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

import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.constants import JobType, MetricsDictKey, Mode
from elasticdl.python.common.log_utils import get_logger
from elasticdl.python.common.model_handler import ModelHandler
from elasticdl.python.common.model_utils import (
    get_dict_from_params_str,
    get_model_spec,
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
        self._init_from_args(args)

    def _init_from_args(self, args):
        """
        Please refer to elastic/python/common/args.py for more
        details about arguments of a worker.
        """
        self._worker_id = args.worker_id
        self._job_type = args.job_type
        self._minibatch_size = args.minibatch_size
        self._log_loss_steps = args.log_loss_steps
        (
            model_inst,
            self._dataset_fn,
            loss,
            opt_fn,
            self._eval_metrics_fn,
            self._prediction_outputs_processor,
            self._custom_data_reader,
            self._callbacks_list,
        ) = get_model_spec(
            model_zoo=args.model_zoo,
            model_def=args.model_def,
            dataset_fn=args.dataset_fn,
            loss=args.loss,
            optimizer=args.optimizer,
            eval_metrics_fn=args.eval_metrics_fn,
            prediction_outputs_processor=args.prediction_outputs_processor,
            custom_data_reader=args.custom_data_reader,
            callbacks=args.callbacks,
        )

        model_handler = ModelHandler.get_model_handler(
            self._distribution_strategy, checkpoint_dir=args.checkpoint_dir
        )
        model_inst = model_handler.get_model_to_train(model_inst)
        model_inst.optimizer = opt_fn()
        model_inst.loss = loss

        self._model_version = -1
        self._task_data_service = TaskDataService(
            self._mc,
            self._job_type == JobType.TRAINING_WITH_EVALUATION,
            custom_data_reader=self._custom_data_reader,
            data_reader_params=get_dict_from_params_str(
                args.data_reader_params
            ),
            data_origin=args.training_data,
        )
        if self._dataset_fn is None:
            if hasattr(
                self._task_data_service.data_reader, "default_dataset_fn"
            ):
                self._dataset_fn = (
                    self._task_data_service.data_reader.default_dataset_fn()
                )
            else:
                raise ValueError(
                    "dataset_fn is required if the data_reader used does "
                    "not provide default implementation of dataset_fn"
                )
        self._get_model_steps = args.get_model_steps
        saved_model_exporter = SavedModelExporter(
            self._task_data_service, self._dataset_fn, model_handler
        )
        # Place default callbacks at the head to execute them firstly
        self._callbacks_list.callbacks.insert(0, saved_model_exporter)
        self._callbacks_list.set_model(model_inst)
        set_callback_parameters(
            self._callbacks_list,
            batch_size=args.minibatch_size,
            saved_model_path=args.output,
            checkpoint_path=args.checkpoint_dir,
        )
        self._saved_model_path = args.output

        if self._distribution_strategy == DistributionStrategy.ALLREDUCE:
            master_addr = args.master_addr.split(":")[0]
            self._trainer = AllReduceTrainer(self._mc, master_addr, model_inst)
        elif (
            self._distribution_strategy
            == DistributionStrategy.PARAMETER_SERVER
        ):
            self._trainer = ParameterServerTrainer(
                model_inst, self._ps_client, self._timing, args
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
            if task_type == elasticdl_pb2.EVALUATION:
                self._trainer.evaluate_minibatch(features, labels)
                break
            elif task_type == elasticdl_pb2.TRAINING:
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
            elif task_type == elasticdl_pb2.PREDICTION:
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

    def _process_eval_task(self, task):
        """
        Check if there are evaluation tasks and process the tasks if any.
        Return:
            A python bool indicating whether worker processed some evaluation
            tasks.
        """
        self.logger.info("the evaluation task_id: %d" % task.task_id)

        gen = self._task_data_service.get_dataset_gen(task)
        if not gen:
            return None

        def create_dataset():
            eval_dataset = tf.data.Dataset.from_generator(
                gen, self._task_data_service.data_reader.records_output_types
            )
            eval_dataset = self._dataset_fn(
                eval_dataset,
                Mode.EVALUATION,
                self._task_data_service.data_reader.metadata,
            )
            eval_dataset = eval_dataset.batch(self._minibatch_size).prefetch(1)
            return eval_dataset

        with tf.device("/device:cpu:0"):
            eval_dataset = create_dataset()
        model_version = task.model_version
        task_id = task.task_id
        err_msg = ""
        for dataset_batch in eval_dataset:
            data_err_msg = self._process_minibatch_and_report(
                dataset_batch, elasticdl_pb2.EVALUATION, model_version
            )
            if data_err_msg:
                err_msg = data_err_msg
                break
        del eval_dataset
        evaluation_result = self._trainer.get_evaluation_result()
        self._mc.report_evaluation_metrics(
            model_outputs=evaluation_result[MetricsDictKey.MODEL_OUTPUT],
            labels=evaluation_result[MetricsDictKey.LABEL],
        )
        self._mc.report_task_result(task_id, err_msg)
        self._trainer.reset_evaluation_result()

    def _process_train_end_callback_task_if_needed(self):
        train_end_task = self._task_data_service.get_train_end_callback_task()
        if train_end_task:
            self._callbacks_list.on_train_end()
            self._task_data_service.clear_train_end_callback_task()
            self._mc.report_task_result(
                task_id=train_end_task.task_id, err_msg=""
            )
        if self._distribution_strategy == DistributionStrategy.ALLREDUCE:
            self._trainer.export_saved_model(self._saved_model_path)

    def _process_minibatch_and_report(
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
        while True:
            dataset = self._task_data_service.get_dataset()
            if not dataset:
                self._process_train_end_callback_task_if_needed()
                break
            dataset = self._dataset_fn(
                dataset,
                Mode.TRAINING,
                self._task_data_service.data_reader.metadata,
            )
            dataset = dataset.batch(self._minibatch_size).prefetch(1)
            self._timing.start_record_time("task_process")
            for dataset_batch in dataset:
                if self._job_type == JobType.TRAINING_WITH_EVALUATION:
                    # Give the worker a chance to process an evaluation task
                    # during training if the task exists
                    evaluation_task_executed = (
                        True
                        if self._evaluate_only()
                        else evaluation_task_executed
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

                err_msg = self._process_minibatch_and_report(
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
        # should not get model before finishing some training tasks, because
        # variables of subclass models are not created.
        while True:
            task = self._mc.get_task(elasticdl_pb2.EVALUATION)
            # no evaluation task in eval_todo of master
            if not task.shard_name:
                break
            self._process_eval_task(task)
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
            dataset = self._dataset_fn(
                dataset,
                Mode.PREDICTION,
                self._task_data_service.data_reader.metadata,
            )
            dataset = dataset.batch(self._minibatch_size).prefetch(1)
            for dataset_batch in dataset:
                task = self._task_data_service.get_current_task()

                err_msg = self._process_minibatch_and_report(
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
            self._train_and_evaluate()
