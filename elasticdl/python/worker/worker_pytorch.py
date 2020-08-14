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
import time
import traceback

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.collective_ops.communicator import CollectiveCommunicator
from elasticdl.python.common.constants import (
    CollectiveCommunicatorStatus,
    Initializer,
    JobType,
    MetricsDictKey,
    Mode,
)
from elasticdl.python.common.dtypes import dtype_numpy_to_tensor
from elasticdl.python.common.log_utils import get_logger
from elasticdl.python.common.model_handler import ModelHandler
from elasticdl.python.common.model_utils import (
    find_layer,
    get_dict_from_params_str,
    get_model_spec,
    get_non_embedding_trainable_vars,
    set_callback_parameters,
)
from elasticdl.python.common.tensor_utils import EmbeddingTableInfo, Tensor
from elasticdl.python.common.timing_utils import Timing
from elasticdl.python.elasticdl.callbacks import SavedModelExporter
from elasticdl.python.elasticdl.feature_column import feature_column
from elasticdl.python.elasticdl.layers.embedding import Embedding
from elasticdl.python.worker.task_data_service import TaskDataService
from elasticdl_client.common.constants import DistributionStrategy

from torch.utils.data import Dataset, DataLoader
import torch

# The default maximum number of a minibatch retry as its results
# (e.g. gradients) are not accepted by master.
DEFAULT_MAX_MINIBATCH_RETRY_NUM = 64

# The default maximum number of retries for allreduce operation
# if allreduce-based distributed training strategy is used.
DEFAULT_MAX_ALLREDUCE_RETRY_NUM = 5
# The default timeout in seconds allowed for reinitializing the
# collective communicator.
DEFAULT_COMMUNICATOR_REINITIALIZING_TIMEOUT = 20


class WorkerPytorch(object):
    """ElasticDL worker"""

    def __init__(
            self,
            args,
            master_client=None,
            ps_client=None,
            max_minibatch_retry_num=DEFAULT_MAX_MINIBATCH_RETRY_NUM,
            max_allreduce_retry_num=DEFAULT_MAX_ALLREDUCE_RETRY_NUM,
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
        self._mc = master_client
        self._ps_client = ps_client
        self._distribution_strategy = args.distribution_strategy
        if (
                self._distribution_strategy
                == DistributionStrategy.PARAMETER_SERVER
        ):
            if self._ps_client is None:
                raise ValueError(
                    "PS channels are not set up under "
                    "parameter server strategy"
                )
            else:
                self._model_versions_from_ps = [
                    -1 for _ in range(self._ps_client.ps_num)
                ]
        self._max_minibatch_retry_num = max_minibatch_retry_num
        self._max_allreduce_retry_num = max_allreduce_retry_num
        self._init_from_args(args)
        self._timing = Timing(args.log_level.upper() == "DEBUG", self.logger)
        self._log_loss_count = 0
        self._var_created = False

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
            self._loss,
            self._opt_fn,
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
            model_params=args.model_params,
            prediction_outputs_processor=args.prediction_outputs_processor,
            custom_data_reader=args.custom_data_reader,
            callbacks=args.callbacks,
        )

        self._collective_communicator = None
        if (
                self._distribution_strategy == DistributionStrategy.ALLREDUCE
                and args.num_workers > 1
        ):
            self._collective_communicator = CollectiveCommunicator(
                service_name=args.collective_communicator_service_name
            )

        self.set_model(model_inst)

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
        self._opt = self._opt_fn()
        self._model.optimizer = self._opt
        self._non_embed_grads = {}
        self._evaluation_result = {}

    def set_model(self, model_inst):
        """Set model instance to worker."""
        self._model = model_inst
        self._model.float()

    def get_model(self):
        self._timing.start_record_time("get_model")
        if (
                self._distribution_strategy
                == DistributionStrategy.PARAMETER_SERVER
        ):
            # 1. Worker tries to pull dense parameters from the PS, maybe one
            # or more PS instances are uninitialized.
            dense_params, uninit_ps = self._ps_client.pull_dense_parameters(
                [i for i in range(self._ps_client.ps_num)],
                self._model_versions_from_ps,
            )

            # 2. Worker pushes local dense parameters to these PS instances
            # to initialize their partition of parameters.
            if len(uninit_ps) > 0:
                for ps_id in uninit_ps:
                    # push variable to ps for initialization
                    parameters = [
                        Tensor(name, self._non_embed_vars[name].numpy(), None)
                        for name in self._ps_client.ps_to_parameter[ps_id]
                    ]
                    self._ps_client.push_dense_parameters(
                        parameters, ps_id, self._model_versions_from_ps[ps_id]
                    )

                ps_params, uninit = self._ps_client.pull_dense_parameters(
                    uninit_ps, self._model_versions_from_ps
                )
                if len(uninit) > 0:
                    # TODO: support PS fault-tolerance
                    raise RuntimeError("PS initialization failed")
                dense_params.update(ps_params)

            # 3. Assign parameters to local model
            for k, v in dense_params.items():
                self._non_embed_vars[k] = torch.from_numpy(v)
            self._model_version = max(self._model_versions_from_ps)
        self._timing.end_record_time("get_model")

    def report_gradient_to_ps(self, gradients):
        """
        report gradient in numpy
        report learning_rate about PyTorch model optimizer
        """
        self._timing.start_record_time("report_gradient")
        grads = []
        for i, v in enumerate(self._non_embed_vars.items()):
            grad = Tensor(v[0], gradients[i].numpy(), None)
            grads.append(grad)
        edl_grads = []

        # TODO: PS in python/go is tf style to get optimizer info,
        #  need PyTorch style, revise when rewrite pserver
        learning_rate = K.get_value(self._model.optimizer.lr)
        # learning_rate = self._model.optimizer.param_groups[0]["lr"]

        accepted, max_version = self._ps_client.push_gradients(
            grads, edl_grads, learning_rate, self._model_versions_from_ps,
        )
        self._timing.end_record_time("report_gradient")
        return accepted, max_version

    def report_gradient(self, grads):
        if (
                self._distribution_strategy
                == DistributionStrategy.PARAMETER_SERVER
        ):
            return self.report_gradient_to_ps(grads)
        else:
            raise RuntimeError(
                "Only support Allreduce and ParameterServer "
                "distribution strategy"
            )

    def _run_model_call_before_training(self, features):
        """
        before training: Create variables and report to ps if not created.
        """
        self._non_embed_vars = {}
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                self._non_embed_vars[name] = param.data
        self._var_created = True

        if self._distribution_strategy == DistributionStrategy.PARAMETER_SERVER:
            self._ps_client.partition_dense_parameters(
                self._non_embed_vars.keys()
            )

    def _collect_gradients_without_allreduce(self, grads):
        accepted, min_model_version = self.report_gradient(grads)
        if accepted and self._get_model_steps > 1:
            non_embed_vars_n = len(self._non_embed_vars)
            self._non_embed_grads = grads[:non_embed_vars_n]
        return accepted, min_model_version

    def training_process_pytorch(self, features, labels):
        outputs = self._model(features)
        loss = self._loss(labels, outputs)
        loss.backward()

        _non_embed_vars_keys = list(self._non_embed_vars.keys())
        grads = []
        grads_tmp = {}
        for name, parms in self._model.named_parameters():
            grads_tmp[name] = parms
        for name in _non_embed_vars_keys:
            if grads_tmp[name].requires_grad:
                grads.append(grads_tmp[name].data)
        return loss, grads

    def _run_training_task(self, features, labels):
        loss, grads = self.training_process_pytorch(features, labels)
        return (*self._collect_gradients_without_allreduce(grads), loss)

    def _process_minibatch(
            self,
            task_type,
            features,
            labels,
            min_model_version,
            train_with_local_model=False,
    ):
        if not self._var_created:
            self._run_model_call_before_training(features)
        self._timing.start_record_time("batch_process")
        for _ in range(self._max_minibatch_retry_num):
            if task_type == elasticdl_pb2.TRAINING:
                # TODO: optimize the logic to avoid unnecessary get_model call.
                if not train_with_local_model:
                    self.get_model()
                *accepted, min_model_version, loss = self._run_training_task(
                    features, labels
                )

                if self._model_version >= self._log_loss_count * self._log_loss_steps:
                    self.logger.info(
                        "Loss = {}, steps = {}".format(
                            loss.numpy(), self._model_version
                        )
                    )
                    self._log_loss_count = (
                            int(self._model_version / self._log_loss_steps) + 1
                    )
                if accepted:
                    break
            elif task_type == elasticdl_pb2.EVALUATION:
                break
            elif task_type == elasticdl_pb2.PREDICTION:
                break
            else:
                raise RuntimeError("Unrecognized task type, %s" % task_type)
        else:
            # Worker got stuck, fail the task.
            # TODO: stop the worker if it fails to make any progress for some time.
            raise RuntimeError("Worker got stuck")
        self._timing.end_record_time("batch_process")
        return min_model_version

    def _process_minibatch_and_report(
            self,
            dataset_batch,
            task_type,
            model_version,
            train_with_local_model=False,
    ):
        """
        train task and report
        """
        err_msg = ""
        try:
            # TODO: dataset is tf style, make it PyTorch style later
            features = dataset_batch[0]
            labels = dataset_batch[1]

            features = torch.from_numpy(features.numpy()).type(torch.FloatTensor)
            labels = torch.from_numpy(labels.numpy()).type(torch.LongTensor)
            features = torch.unsqueeze(features, dim=1)

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

    def _train(self):
        """
        Train the model on the worker
        _dataset_fn() is based on TensorFlow func, deal with the RecordIO data
        TODO: rewrite _dataset_fn() to support PyTorch dataset
        """

        train_with_local_model = False
        while True:
            dataset = self._task_data_service.get_dataset()
            if not dataset:
                break
            dataset = self._dataset_fn(
                dataset,
                Mode.TRAINING,
                self._task_data_service.data_reader.metadata,
            )
            self._timing.start_record_time("task_process")
            for dataset_batch in dataset:
                task = self._task_data_service.get_current_task()
                err_msg = self._process_minibatch_and_report(
                    dataset_batch,
                    task.type,
                    task.model_version,
                    train_with_local_model,
                )
                if self._task_data_service.report_record_done(
                        self._minibatch_size, err_msg
                ):
                    self._timing.end_record_time("task_process")
                    self._timing.report_timing(reset=True)
                    self._timing.start_record_time("task_process")
            del dataset

    def run(self):
        """
        Fetches task from master with and train
        """
        self._train()
