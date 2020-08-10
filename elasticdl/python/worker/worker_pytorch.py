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


class CustomDataset(torch.utils.data.IterableDataset):
    def __init__(self, data):
        self.data_source = data

    def __iter__(self):
        return iter(self.data_source)


class Worker_pytorch(object):
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
        self._opt = self._opt_fn(model_inst)
        self._model.optimizer = self._opt
        self._non_embed_grads = {}
        self._evaluation_result = {}

    # TODO: Multiple tests are currently using this function to initialize
    # self._model, where the initialization should be done via constructor.
    def set_model(self, model_inst):
        """Set model instance to worker."""
        self._model = model_inst
        self._model.float()
        self._train_eagerly = False
        # self._init_embeddings()

    def _init_embedding_layer(self):
        """
        Init elasticdl.layers.embedding layer list and assign worker to them
        """
        self._embedding_layers = find_layer(self._model, Embedding)
        if (
                self._distribution_strategy
                == DistributionStrategy.PARAMETER_SERVER
        ):
            for layer in self._embedding_layers:
                layer.set_lookup_embedding_func(
                    self._ps_client.pull_embedding_vectors
                )

    def _init_embedding_column(self):
        self._embedding_columns = []
        for layer in self._model.layers:
            if isinstance(layer, tf.keras.layers.DenseFeatures):
                for column in layer._feature_columns:
                    if isinstance(column, feature_column.EmbeddingColumn):
                        self._embedding_columns.append(column)
                        self.logger.info(
                            "Initialize ElasticDL EmbeddingColumn:{}".format(
                                column.name
                            )
                        )

        if (
                self._distribution_strategy
                == DistributionStrategy.PARAMETER_SERVER
        ):
            for column in self._embedding_columns:
                column.set_lookup_embedding_func(
                    self._ps_client.pull_embedding_vectors
                )

    def _check_name_conflict_of_embedding_layer_and_column(self):
        if not self._embedding_layers or not self._embedding_columns:
            return

        embedding_layer_name_set = set(
            [layer.name for layer in self._embedding_layers]
        )
        embedding_column_name_set = set(
            [column.name for column in self._embedding_columns]
        )
        conflict_name_set = embedding_column_name_set.union(
            embedding_layer_name_set
        )
        if conflict_name_set:
            raise Exception(
                "Name conflict between embedding layer and column: {}".format(
                    conflict_name_set
                )
            )

    def _update_local_model(self):
        if not self._non_embed_grads:
            return
        # Take care of the order of grads and vars if worker modifies
        # `_non_embed_vars` during training.
        self._opt.apply_gradients(
            zip(self._non_embed_grads, self._non_embed_vars.values())
        )
        self._non_embed_grads = None

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
                # print("type(v):",type(v))
                # self._non_embed_vars[k].assign(v)
                self._non_embed_vars[k] = torch.from_numpy(v)

            self._model_version = max(self._model_versions_from_ps)
        self._timing.end_record_time("get_model")

    def report_embedding_info(self):
        # TODO(qijun): only support float32
        infos = []
        if self._embedding_layers:
            for layer in self._embedding_layers:
                infos.append(
                    EmbeddingTableInfo(
                        layer.embedding_weight_name,
                        layer.output_dim,
                        layer.embeddings_initializer,
                        dtype_numpy_to_tensor(np.dtype("float32")),
                    )
                )

        if self._embedding_columns:
            for column in self._embedding_columns:
                # TODO(brightcoder01): The initializer in embedding column is
                # a variable initializer function. For embedding layer, it's a
                # tf.keras.initializers. Keep aligned between these two.
                infos.append(
                    EmbeddingTableInfo(
                        column.embedding_weight_name,
                        column.dimension,
                        Initializer.UNIFORM,
                        dtype_numpy_to_tensor(np.dtype("float32")),
                    )
                )

        self._ps_client.push_embedding_table_infos(infos)

    def _collect_edl_embedding_name_values(self):
        """
        Collect the information of ElasticDL customized
        embeddings such as EDL embedding layer and EDL embedding column.
        Return:
            An array of key-value pair.
            Key is embedding names, layer name for embedding layer
            and column name for embedding column.
            Value is the EmbeddingAndIds tuple.
        """
        embedding_name_values = []
        return embedding_name_values

    def report_gradient_to_ps(self, gradients):
        self._timing.start_record_time("report_gradient")
        grads = []
        for i, v in enumerate(self._non_embed_vars.items()):
            grad = Tensor(v[0], gradients[i].numpy(), None)
            grads.append(grad)
        edl_grads = []
        edl_embedding_name_values = self._collect_edl_embedding_name_values()
        if edl_embedding_name_values:
            non_embed_vars_n = len(self._non_embed_vars)
            edl_embedding_grads = gradients[non_embed_vars_n:]
            bet_number = 0
            for name, embedding_and_ids in edl_embedding_name_values:

                for i in range(bet_number):
                    grad = Tensor(
                        name,
                        edl_embedding_grads[i + bet_number].values.numpy(),
                        edl_embedding_grads[i + bet_number].indices.numpy(),
                    )
                    edl_grads.append(grad)
                bet_number += len(embedding_and_ids)
            if len(edl_embedding_grads) != bet_number:
                raise ValueError(
                    "elasticdl.layers.embedding related gradient number %d "
                    "does not match the number of its output tensor %d."
                    % (len(edl_embedding_grads), bet_number)
                )
        # learning_rate = K.get_value(self._model.optimizer.lr)
        learning_rate = np.float32(0.001)
        accepted, max_version = self._ps_client.push_gradients(
            grads, edl_grads, learning_rate, self._model_versions_from_ps,
        )
        self._timing.end_record_time("report_gradient")
        return accepted, max_version

    def report_gradient_locally(self, grads):
        if self._embedding_layers or self._embedding_columns:
            raise ValueError(
                "ElasticDL embedding layer is not supported when"
                "reporting gradients locally"
            )
        self._non_embed_grads = grads[: len(self._non_embed_vars)]
        return True, None

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
        """Call `self._model.call` before training for two things:
            * Create variables and report to ps if not created.
            * Check whether there is an embedding layer that is called
              more than once during one forward-pass.
        """
        print("features.dtype:", features.dtype)  # torch.int64
        _ = self._model.forward(features)

        self._non_embed_vars = {}
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                self._non_embed_vars[name] = param.data

        self._var_created = True

        if (
                self._distribution_strategy
                == DistributionStrategy.PARAMETER_SERVER
        ):
            self._ps_client.partition_dense_parameters(
                self._non_embed_vars.keys()
            )

    def get_trainable_items(self):
        """
        return all trainable variables list, including batch embedding
        tensor (BET) if exists. take care to keep the same order as in
        self.report_gradient()
        """
        bets = []
        if self._embedding_layers:
            for layer in self._embedding_layers:
                bets.extend(
                    [
                        batch_embedding
                        for (batch_embedding, _) in layer.embedding_and_ids
                    ]
                )

        if self._embedding_columns:
            for column in self._embedding_columns:
                bets.extend(
                    [
                        batch_embedding
                        for (batch_embedding, _) in column.embedding_and_ids
                    ]
                )

        return list(self._non_embed_vars.values()) + bets

    def _collect_gradients_without_allreduce(self, grads):
        accepted, min_model_version = self.report_gradient(grads)
        if accepted and self._get_model_steps > 1:
            non_embed_vars_n = len(self._non_embed_vars)
            self._non_embed_grads = grads[:non_embed_vars_n]
        self._reset_embedding()
        return accepted, min_model_version

    def training_process_pytorch(self, features, labels):
        outputs = self._model(features)
        print("training_process_pytorch:")
        print("type(outputs):", type(outputs), outputs.size())
        print("type(labels):", type(labels), labels.size())
        loss = self._loss(labels, outputs)
        loss.backward()

        _non_embed_vars_keys = list(self._non_embed_vars.keys())
        grads = []
        loss_tmp = {}
        for name, parms in self._model.named_parameters():
            loss_tmp[name] = parms
        for name in _non_embed_vars_keys:
            if loss_tmp[name].requires_grad:
                grads.append(loss_tmp[name].data)
        return loss, grads

    def _run_training_task(self, features, labels):
        loss, grads = self.training_process_pytorch(features, labels)
        if self._distribution_strategy == DistributionStrategy.ALLREDUCE:
            # TODO: Delay certain amount of time before retrying
            for _ in range(self._max_allreduce_retry_num + 1):
                accepted = self._collect_gradients_with_allreduce_robust(grads)
                if accepted:
                    return accepted, None, loss
                else:
                    self.logger.warning(
                        "Failed to perform allreduce operation on"
                        "the gradients. Retrying..."
                    )
        else:
            return (*self._collect_gradients_without_allreduce(grads), loss)

    def _process_minibatch(
            self,
            task_type,
            features,
            labels,
            min_model_version,
            train_with_local_model=False,
    ):
        if self._need_embedding_layer_check or not self._var_created:
            self._run_model_call_before_training(features)
        self._timing.start_record_time("batch_process")
        for _ in range(self._max_minibatch_retry_num):
            if task_type == elasticdl_pb2.EVALUATION:
                break
            elif task_type == elasticdl_pb2.TRAINING:
                # TODO: optimize the logic to avoid unnecessary
                #       get_model call.
                if not train_with_local_model:
                    self.get_model()
                self._callbacks_list.on_train_batch_begin(self._model_version)
                *accepted, min_model_version, loss = self._run_training_task(
                    features, labels
                )
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
                if accepted:
                    break
            elif task_type == elasticdl_pb2.PREDICTION:
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

    def _process_train_end_callback_task_if_needed(self):
        train_end_task = self._task_data_service.get_train_end_callback_task()
        if train_end_task:
            self._callbacks_list.on_train_end()
            self._task_data_service.clear_train_end_callback_task()
            self._mc.report_task_result(
                task_id=train_end_task.task_id, err_msg=""
            )

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

    def _dataset_pytorch(self, dataset, minibatch_size):
        dataset = list(dataset.as_numpy_iterator())
        iterable_dataset = CustomDataset(dataset)
        dataloader = DataLoader(dataset=iterable_dataset, batch_size=minibatch_size)
        return dataloader

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
            # dataset = dataset.batch(self._minibatch_size).prefetch(1)
            dataset = self._dataset_pytorch(dataset, self._minibatch_size)
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

    def run(self):
        """
        Fetches task from master with and performs training, evaluation
        or prediction.
        """
        self._train_and_evaluate()
