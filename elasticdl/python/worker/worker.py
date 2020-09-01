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

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.constants import (
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
from elasticdl.python.worker.allreduce_trainer import AllReduceTrainer
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
            prediction_outputs_processor=args.prediction_outputs_processor,
            custom_data_reader=args.custom_data_reader,
            callbacks=args.callbacks,
        )

        self._model_handler = ModelHandler.get_model_handler(
            self._distribution_strategy, checkpoint_dir=args.checkpoint_dir
        )
        model_inst = self._model_handler.get_model_to_train(model_inst)
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

        saved_model_exporter = SavedModelExporter(
            self._task_data_service, self._dataset_fn, self._model_handler
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

        self._allreduce_trainer = None
        if self._distribution_strategy == DistributionStrategy.ALLREDUCE:
            master_addr = args.master_addr.split(":")[0]
            self._allreduce_trainer = AllReduceTrainer(
                self._mc, master_addr, self._model, self._loss, self._opt
            )

    # TODO: Multiple tests are currently using this function to initialize
    # self._model, where the initialization should be done via constructor.
    def set_model(self, model_inst):
        """Set model instance to worker."""
        self._model = model_inst
        self._train_eagerly = False
        self._init_embeddings()

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

    def _init_embeddings(self):
        self._init_embedding_layer()
        self._init_embedding_column()
        self._check_name_conflict_of_embedding_layer_and_column()

        if (
            self._distribution_strategy
            == DistributionStrategy.PARAMETER_SERVER
        ):
            self.report_embedding_info()

        self._need_embedding_layer_check = (
            True
            if self._embedding_layers or self._embedding_columns
            else False
        )

    def _set_tape_for_embedding(self, tape):
        for layer in self._embedding_layers:
            layer.set_tape(tape)
        for column in self._embedding_columns:
            column.set_tape(tape)

    def _reset_embedding(self):
        for layer in self._embedding_layers:
            layer.reset()
        for column in self._embedding_columns:
            column.reset()

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
                self._non_embed_vars[k].assign(v)

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
        for layer in self._embedding_layers:
            embedding_name_values.append(
                (layer.embedding_weight_name, layer.embedding_and_ids)
            )
        for column in self._embedding_columns:
            embedding_name_values.append(
                (column.embedding_weight_name, column.embedding_and_ids)
            )

        return embedding_name_values

    def report_gradient(self, gradients):
        self._timing.start_record_time("report_gradient")

        grads = []
        for i, v in enumerate(self._non_embed_vars.values()):
            if isinstance(gradients[i], tf.IndexedSlices):
                grad = Tensor(
                    v.name,
                    gradients[i].values.numpy(),
                    gradients[i].indices.numpy(),
                )
            else:
                grad = Tensor(v.name, gradients[i].numpy(), None)
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
        learning_rate = K.get_value(self._model.optimizer.lr)
        accepted, max_version = self._ps_client.push_gradients(
            grads, edl_grads, learning_rate, self._model_versions_from_ps,
        )
        self._timing.end_record_time("report_gradient")
        return accepted, max_version

    def report_prediction_outputs(self, predictions):
        if self._prediction_outputs_processor:
            self._prediction_outputs_processor.process(
                predictions, self._worker_id
            )
        else:
            self.logger.warning(
                "prediction_outputs_processor is not "
                "defined in the model definition. Prediction outputs "
                "are not processed."
            )
        return True

    def _run_model_call_before_training(self, features):
        """Call `self._model.call` before training for two things:
            * Create variables and report to ps if not created.
            * Check whether there is an embedding layer that is called
              more than once during one forward-pass.
        """
        if self._embedding_layers:
            with tf.GradientTape() as tape:
                self._set_tape_for_embedding(tape)
                _ = self._model.call(features)
        else:
            _ = self._model.call(features)
        self._non_embed_vars = {}
        for var in get_non_embedding_trainable_vars(
            self._model, self._embedding_layers
        ):
            self._non_embed_vars[var.name] = var

        self._var_created = True

        if (
            self._distribution_strategy
            == DistributionStrategy.PARAMETER_SERVER
        ):
            self._ps_client.partition_dense_parameters(
                self._non_embed_vars.keys()
            )

        if self._need_embedding_layer_check:
            self._train_eagerly = False
            for layer in self._embedding_layers:
                if len(layer.embedding_and_ids) > 1:
                    self._train_eagerly = True
                    self.logger.warning(
                        "ElasticDL embedding layer %s is called more than "
                        "once, this will make the training process unable "
                        "to accelerate with tf.function." % (layer.name)
                    )
            self._need_embedding_layer_check = False

        self._reset_embedding()

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

    def training_process(self, features, labels):
        """
        training for models with elasticdl.layers.embedding does not
        support tf.function decorator
        """
        if self._train_eagerly:
            return self.training_process_eagerly(features, labels)
        else:
            return self.training_process_with_acceleration(features, labels)

    @tf.function
    def training_process_with_acceleration(self, features, labels):
        return self.training_process_eagerly(features, labels)

    def training_process_eagerly(self, features, labels):
        with tf.GradientTape() as tape:
            self._set_tape_for_embedding(tape)
            outputs = self._model.call(features, training=True)
            loss = self._loss(labels, outputs)
            # Add regularization loss if any
            if self._model.losses:
                loss += tf.math.add_n(self._model.losses)
        grads = tape.gradient(loss, self.get_trainable_items())
        return loss, grads

    @tf.function
    def forward_process(self, features):
        """Calculates model outputs in non-training mode."""
        outputs = self._model.call(features, training=False)
        return outputs

    def _get_local_model_params(self):
        return [v for v in self._non_embed_vars.values()]

    def _collect_gradients_with_ps(self, grads):
        accepted, min_model_version = self.report_gradient(grads)
        if accepted and self._get_model_steps > 1:
            non_embed_vars_n = len(self._non_embed_vars)
            self._non_embed_grads = grads[:non_embed_vars_n]
        self._reset_embedding()
        return accepted, min_model_version

    def _run_training_task(self, features, labels):
        if self._distribution_strategy == DistributionStrategy.ALLREDUCE:
            (
                version,
                loss,
            ) = self._allreduce_trainer.training_process_with_fault_tolerance(
                features, labels
            )
            self._model_version = version
            return True, version, loss
        else:
            loss, grads = self.training_process(features, labels)
            return (*self._collect_gradients_with_ps(grads), loss)

    def _collect_evaluation_result(self, outputs, labels):
        key = MetricsDictKey.MODEL_OUTPUT
        if key not in self._evaluation_result:
            outputs = {k: [v.numpy()] for k, v in outputs.items()}
            self._evaluation_result[key] = outputs
        else:
            for k, v in outputs.items():
                self._evaluation_result[key][k].append(v.numpy())
        key = MetricsDictKey.LABEL
        if key not in self._evaluation_result:
            self._evaluation_result[key] = [labels.numpy()]
        else:
            self._evaluation_result[key].append(labels.numpy())

    def _run_evaluation_task(self, features, labels):
        outputs = self.forward_process(features)
        if not isinstance(outputs, dict):
            outputs = {MetricsDictKey.MODEL_OUTPUT: outputs}
        self._collect_evaluation_result(outputs, labels)

    def _run_prediction_task(self, features):
        predictions = self.forward_process(features)
        return self.report_prediction_outputs(predictions)

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
                self._run_evaluation_task(features, labels)
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
                if self._model_version != min_model_version:
                    self.get_model()
                accepted = self._run_prediction_task(features)
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
        self._mc.report_evaluation_metrics(
            model_outputs=self._evaluation_result[MetricsDictKey.MODEL_OUTPUT],
            labels=self._evaluation_result[MetricsDictKey.LABEL],
        )
        self._mc.report_task_result(task_id, err_msg)
        self._evaluation_result = {}

    def _process_train_end_callback_task_if_needed(self):
        train_end_task = self._task_data_service.get_train_end_callback_task()
        if train_end_task:
            self._callbacks_list.on_train_end()
            self._task_data_service.clear_train_end_callback_task()
            self._mc.report_task_result(
                task_id=train_end_task.task_id, err_msg=""
            )
        if self._distribution_strategy == DistributionStrategy.ALLREDUCE:
            self._allreduce_trainer.export_saved_model(self._saved_model_path)

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
            if self._allreduce_trainer:
                self._allreduce_trainer.init_horovod_if_needed()
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

                if (
                    self._allreduce_trainer
                    and self._model_version % DEFAULT_STEPS_TO_CHECK_RENDEZVOUS
                    == 0
                ):
                    self._allreduce_trainer.init_horovod_if_needed()

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
        is_model_got = False
        while True:
            task = self._mc.get_task(elasticdl_pb2.EVALUATION)
            # no evaluation task in eval_todo of master
            if not task.shard_name:
                break
            # get the latest model before processing eval tasks
            if not is_model_got:
                self.get_model()
                is_model_got = True
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
