import os
import time
import traceback

import numpy as np
import tensorflow as tf
from google.protobuf import empty_pb2

from elasticdl.proto import elasticdl_pb2, elasticdl_pb2_grpc
from elasticdl.python.collective_ops.communicator import CollectiveCommunicator
from elasticdl.python.common.constants import (
    CollectiveCommunicatorStatus,
    DistributionStrategy,
    JobType,
    MetricsDictKey,
    Mode,
    SaveModelConfig,
)
from elasticdl.python.common.hash_utils import (
    int_to_id,
    scatter_embedding_vector,
    string_to_id,
)
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.common.model_handler import ModelHandler
from elasticdl.python.common.model_utils import (
    find_layer,
    get_dict_from_params_str,
    get_model_spec,
    get_non_embedding_trainable_vars,
)
from elasticdl.python.common.tensor import (
    Tensor,
    emplace_tensor_pb_from_ndarray,
    serialize_tensor,
    tensor_pb_to_ndarray,
)
from elasticdl.python.elasticdl.layers.embedding import Embedding
from elasticdl.python.worker.task_data_service import TaskDataService

# The default maximum number of a minibatch retry as its results
# (e.g. gradients) are not accepted by master.
DEFAULT_MAX_MINIBATCH_RETRY_NUM = 64

# The default maximum number of retries for allreduce operation
# if allreduce-based distributed training strategy is used.
DEFAULT_MAX_ALLREDUCE_RETRY_NUM = 5


class Worker(object):
    """ElasticDL worker"""

    def __init__(
        self,
        args,
        channel=None,
        ps_channels=None,
        max_minibatch_retry_num=DEFAULT_MAX_MINIBATCH_RETRY_NUM,
        max_allreduce_retry_num=DEFAULT_MAX_ALLREDUCE_RETRY_NUM,
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
        if channel is None:
            self._stub = None
        else:
            self._stub = elasticdl_pb2_grpc.MasterStub(channel)

        self._use_multi_ps = False
        if isinstance(ps_channels, list):
            if len(ps_channels) > 0:
                self._use_multi_ps = True
                self._ps_stubs = [
                    elasticdl_pb2_grpc.PserverStub(c) for c in ps_channels
                ]
                self._var_to_ps = {}
        self._distribution_strategy = args.distribution_strategy
        if (
            self._distribution_strategy
            == DistributionStrategy.PARAMETER_SERVER
            and self._use_multi_ps is False
        ):
            raise ValueError(
                "PS channels are not set up under parameter server strategy"
            )

        self._max_minibatch_retry_num = max_minibatch_retry_num
        self._max_allreduce_retry_num = max_allreduce_retry_num
        self._init_from_args(args)

    def _init_from_args(self, args):
        """
        Please refer to elastic/python/common/args.py for more
        details about arguments of a worker.
        """
        self._worker_id = args.worker_id
        self._job_type = args.job_type
        self._minibatch_size = args.minibatch_size
        (
            model_inst,
            self._dataset_fn,
            self._loss,
            self._opt_fn,
            self._eval_metrics_fn,
            self._prediction_outputs_processor,
        ) = get_model_spec(
            model_zoo=args.model_zoo,
            model_def=args.model_def,
            dataset_fn=args.dataset_fn,
            loss=args.loss,
            optimizer=args.optimizer,
            eval_metrics_fn=args.eval_metrics_fn,
            model_params=args.model_params,
            prediction_outputs_processor=args.prediction_outputs_processor,
        )

        self._collective_communicator = (
            CollectiveCommunicator()
            if self._distribution_strategy == DistributionStrategy.ALLREDUCE
            else None
        )
        self._model_handler = ModelHandler.get_model_handler(
            self._distribution_strategy, checkpoint_dir=args.checkpoint_dir
        )
        model_inst = self._model_handler.get_model_to_train(model_inst)
        self.set_model(model_inst)

        self._model_version = -1
        self._task_data_service = TaskDataService(
            self,
            self._job_type == JobType.TRAINING_WITH_EVALUATION,
            data_reader_params=get_dict_from_params_str(
                args.data_reader_params
            ),
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
        if self._get_model_steps > 1:
            self._opt = self._opt_fn()
        self._non_embed_grads = {}
        self._evaluation_result = {}

    # TODO: Multiple tests are currently using this function to initialize
    # self._model, where the initialization should be done via constructor.
    def set_model(self, model_inst):
        """Set model instance to worker."""
        self._model = model_inst
        self._train_eagerly = False
        self._init_embedding_layer()
        self._var_created = self._model.built
        self._non_embed_vars = {}
        if self._var_created:
            for var in get_non_embedding_trainable_vars(
                self._model, self._embedding_layers
            ):
                self._non_embed_vars[var.name] = var
            if self._use_multi_ps:
                self.init_ps_var_partition()

    def _init_embedding_layer(self):
        """
        Init elasticdl.layers.embedding layer list and assign worker to them
        """
        self._embedding_layers = find_layer(self._model, Embedding)
        if self._use_multi_ps:
            self.report_embedding_info()
            for layer in self._embedding_layers:
                layer.set_lookup_embedding_func(self.pull_embedding_vector)

        self._need_embedding_layer_check = (
            True if self._embedding_layers else False
        )

    def _set_tape_for_embedding(self, tape):
        for layer in self._embedding_layers:
            layer.set_tape(tape)

    def _reset_embedding(self):
        for layer in self._embedding_layers:
            layer.reset()

    def _update_local_model(self):
        if not self._non_embed_grads:
            return
        # Take care of the order of grads and vars if worker modifies
        # `_non_embed_vars` during training.
        self._opt.apply_gradients(
            zip(self._non_embed_grads, self._non_embed_vars.values())
        )
        self._non_embed_grads = None

    def get_task(self, task_type=None):
        """
        get task from master
        """
        req = elasticdl_pb2.GetTaskRequest()
        req.worker_id = self._worker_id
        if task_type is not None:
            req.task_type = task_type

        return self._stub.get_task(req)

    def get_model(self):
        model_version = -1
        variable_future_and_id_pairs = []
        req = empty_pb2.Empty()
        for ps_id, stub in enumerate(self._ps_stubs):
            if ps_id not in self._ps_vars:
                continue
            # async grpc call
            var_future = stub.pull_variable.future(req)
            variable_future_and_id_pairs.append((var_future, ps_id))

        for var_future, ps_id in variable_future_and_id_pairs:
            res = var_future.result()
            if not res.model_init_status:
                # push variable to ps for initialization
                self.report_variable_to_ps(ps_id)
                res = self._ps_stubs[ps_id].pull_variable(req)
                if not res.model_init_status:
                    # TODO: support PS fault-tolerance
                    raise RuntimeError(
                        "PS pod %d cannot be initialized" % ps_id
                    )

            for tensor_pb in res.model.param:
                tensor = Tensor.from_tensor_pb(tensor_pb)
                self._non_embed_vars[tensor.name].assign(tensor.to_ndarray())

            model_version = max(model_version, res.model.version)
        self._model_version = model_version

    def pull_embedding_vector(self, layer_name, embedding_ids):
        """Pulls and returns embedding vectors ordered by the embedding ids."""
        ps_ids = {}
        ps_ids_index = {}
        for idx, embedding_id in enumerate(embedding_ids):
            ps_id = int_to_id(embedding_id, len(self._ps_stubs))
            ps_ids.setdefault(ps_id, []).append(embedding_id)
            ps_ids_index.setdefault(ps_id, []).append(idx)

        embeddings = []
        index = []
        pb_future_and_id_pairs = []
        for ps_id, embedding_ids in ps_ids.items():
            req = elasticdl_pb2.PullEmbeddingVectorRequest()
            req.name = layer_name
            req.ids.extend(embedding_ids)
            pb_future = self._ps_stubs[ps_id].pull_embedding_vector.future(req)
            pb_future_and_id_pairs.append((pb_future, ps_id))
        for pb_future, ps_id in pb_future_and_id_pairs:
            pb = pb_future.result()
            embeddings.append(tensor_pb_to_ndarray(pb))
            index.extend(ps_ids_index[ps_id])
        embeddings = np.concatenate(embeddings)

        # adjust the order of embedding vectors
        new_embeddings = np.empty_like(embeddings)
        new_embeddings[index] = embeddings
        return new_embeddings

    def report_task_result(self, task_id, err_msg, exec_counters=None):
        """
        report task result to master
        """
        report = elasticdl_pb2.ReportTaskResultRequest()
        report.task_id = task_id
        report.err_message = err_msg
        if isinstance(exec_counters, dict):
            report.exec_counters.update(exec_counters)
        return self._stub.report_task_result(report)

    def init_ps_var_partition(self):
        ps_vars = {}
        for v in self._non_embed_vars.values():
            if v.name not in self._var_to_ps:
                self._var_to_ps[v.name] = string_to_id(
                    v.name, len(self._ps_stubs)
                )
            ps_id = self._var_to_ps[v.name]
            if ps_id not in ps_vars:
                ps_vars[ps_id] = [v]
            else:
                ps_vars[ps_id].append(v)
        self._ps_vars = ps_vars

    def report_embedding_info(self):
        model = elasticdl_pb2.Model()
        if self._embedding_layers:
            embedding_infos = model.embedding_table_info
            for layer in self._embedding_layers:
                embedding_info = embedding_infos.add()
                embedding_info.name = layer.name
                embedding_info.dim = layer.output_dim
                embedding_info.initializer = layer.embeddings_initializer

        for ps_id in range(len(self._ps_stubs)):
            self._ps_stubs[ps_id].push_embedding_info(model)

    def report_variable_to_ps(self, ps_id):
        model = elasticdl_pb2.Model()
        if ps_id in self._ps_vars:
            vars = self._ps_vars[ps_id]
            for var in vars:
                emplace_tensor_pb_from_ndarray(
                    model.param, var.numpy(), name=var.name
                )
        self._ps_stubs[ps_id].push_model(model)

    def report_variable(self):
        # TODO: call `push_model` in parallel
        for ps_id in range(len(self._ps_stubs)):
            self.report_variable_to_ps(ps_id)

    def report_gradient_to_ps(self, grads):
        reqs = [
            elasticdl_pb2.PushGradientRequest()
            for i in range(len(self._ps_stubs))
        ]
        ps_grads = {}
        non_embed_vars_n = len(self._non_embed_vars)
        for g, v in zip(
            grads[:non_embed_vars_n], self._non_embed_vars.values()
        ):
            ps_id = self._var_to_ps[v.name]
            if ps_id not in ps_grads:
                ps_grads[ps_id] = [(g, v.name)]
            else:
                ps_grads[ps_id].append((g, v.name))

        for ps_id in ps_grads:
            req = reqs[ps_id]
            for g, name in ps_grads[ps_id]:
                emplace_tensor_pb_from_ndarray(req.gradients, g, name=name)

        if self._embedding_layers:
            edl_embedding_grads = grads[non_embed_vars_n:]
            bet_number = 0
            for layer in self._embedding_layers:
                bet_number += len(layer.embedding_and_ids)
            if len(edl_embedding_grads) != bet_number:
                raise ValueError(
                    "elasticdl.layers.embedding related gradient number %d "
                    "does not match the number of its output tensor %d."
                    % (len(edl_embedding_grads), bet_number)
                )

            grad_accum_iter = 0
            for layer in self._embedding_layers:
                g_values = None
                g_indices = None
                for _, ids in layer.embedding_and_ids:
                    grad = edl_embedding_grads[grad_accum_iter]
                    grad_accum_iter += 1
                    # ElasticDL embedding layer with Sparse Gradients
                    if isinstance(grad, tf.IndexedSlices):
                        grad = grad.values
                    if g_values is not None:
                        g_values = tf.concat([g_values, grad], axis=0)
                        g_indices = tf.concat([g_indices, ids], axis=0)
                    else:
                        g_values = grad
                        g_indices = ids

                results = scatter_embedding_vector(
                    g_values.numpy(), g_indices.numpy(), len(self._ps_stubs)
                )

                for ps_id in results:
                    req = reqs[ps_id]
                    gv, gi = results[ps_id]
                    emplace_tensor_pb_from_ndarray(
                        req.gradients, values=gv, indices=gi, name=layer.name
                    )

        report_futures = []
        for ps_id in range(len(self._ps_stubs)):
            req = reqs[ps_id]
            req.model_version = self._model_version
            report_future = self._ps_stubs[ps_id].push_gradient.future(req)
            report_futures.append(report_future)

        accepted = False
        max_version = -1
        for report_future in report_futures:
            res = report_future.result()
            if res.accepted:
                accepted = True
            if res.model_version > max_version:
                max_version = res.model_version
        return accepted, max_version

    def report_gradient_locally(self, grads):
        if self._embedding_layers:
            raise ValueError(
                "ElasticDL embedding layer is not supported when"
                "reporting gradients locally"
            )
        for g, v in zip(
            grads[: len(self._non_embed_vars)], self._non_embed_vars.values()
        ):
            self._non_embed_grads[v.name] = v
        return True, None

    def report_gradient(self, grads):
        if self._distribution_strategy == DistributionStrategy.ALLREDUCE:
            return self.report_gradient_locally(grads)
        else:
            if self._use_multi_ps:
                return self.report_gradient_to_ps(grads)
            raise RuntimeError("Only support report gradients to PS")

    def report_evaluation_metrics(self, model_outputs, labels):
        """
        report evaluation metrics to ps.
        """
        req = elasticdl_pb2.ReportEvaluationMetricsRequest()
        for name, output in model_outputs.items():
            output = np.concatenate(output)
            emplace_tensor_pb_from_ndarray(
                req.model_outputs, output, name=name
            )
        labels = np.concatenate(labels)
        tensor = Tensor(values=labels)
        serialize_tensor(tensor, req.labels)
        self._stub.report_evaluation_metrics(req)

    def report_prediction_outputs(self, predictions):
        if self._prediction_outputs_processor:
            self._prediction_outputs_processor.process(
                predictions, self._worker_id
            )
        else:
            logger.warning(
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

        if not self._var_created:
            if self._use_multi_ps:
                self.init_ps_var_partition()
            else:
                self.report_variable()
            self._var_created = True

        if self._need_embedding_layer_check:
            self._train_eagerly = False
            for layer in self._embedding_layers:
                if len(layer.embedding_and_ids) > 1:
                    self._train_eagerly = True
                    logger.warning(
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

    # TODO: Reuse model handler to initialize the model parameters properly
    def _update_local_model_params(self, model_params):
        pass

    def _get_local_model_params(self):
        return {}

    # TODO: Implement master gRPC service to select a worker
    # to be used for broadcast model parameters from.
    def _get_broadcast_root_worker_ip(self):
        return 1

    def _broadcast_model_params(self):
        status = self._collective_communicator.barrier()
        if status == CollectiveCommunicatorStatus.FAILED:
            logger.warning("Failed to perform barrier operation")
            return False
        status, model_params = self._collective_communicator.broadcast(
            self._get_local_model_params(),
            self._get_broadcast_root_worker_ip(),
        )
        if status == CollectiveCommunicatorStatus.FAILED:
            logger.warning("Failed to broadcast model parameters")
            return False
        self._update_local_model_params(model_params)
        status = self._collective_communicator.barrier()
        if status == CollectiveCommunicatorStatus.FAILED:
            logger.warning("Failed to perform barrier operation")
            return False
        return True

    def _calculate_grads_and_report_with_allreduce(self, grads):
        status, averaged_grads = self._collective_communicator.allreduce(grads)
        accepted = False
        if status == CollectiveCommunicatorStatus.SUCCEEDED:
            accepted, _ = self.report_gradient(averaged_grads)
            if not accepted:
                logger.warning("Failed to report the averaged gradients")
        return accepted

    def _collect_gradients_with_allreduce_robust(self, grads):
        accepted = self._calculate_grads_and_report_with_allreduce(grads)
        if not accepted:
            if self._collective_communicator.has_new_worker_joining():
                succeeded = self._broadcast_model_params()
                if succeeded:
                    return self._calculate_grads_and_report_with_allreduce(
                        grads
                    )
                else:
                    return False
            else:
                logger.warning(
                    "No new worker joining. Broadcast operation skipped"
                )
                return False
        else:
            return True

    def _collect_gradients_without_allreduce(self, grads):
        accepted, min_model_version = self.report_gradient(grads)
        if accepted and self._get_model_steps > 1:
            non_embed_vars_n = len(self._non_embed_vars)
            self._non_embed_grads = grads[:non_embed_vars_n]
        self._reset_embedding()
        return accepted, min_model_version

    def _run_training_task(self, features, labels):
        loss, grads = self.training_process(features, labels)
        if self._distribution_strategy == DistributionStrategy.ALLREDUCE:
            # TODO: Delay certain amount of time before retrying
            for _ in range(self._max_allreduce_retry_num + 1):
                accepted = self._collect_gradients_with_allreduce_robust(grads)
                if accepted:
                    return accepted, None, loss
                else:
                    logger.warning(
                        "Failed to perform allreduce operation on"
                        "the gradients. Retrying..."
                    )
        else:
            return (*self._collect_gradients_without_allreduce(grads), loss)

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
        for _ in range(self._max_minibatch_retry_num):
            if task_type == elasticdl_pb2.EVALUATION:
                self._run_evaluation_task(features, labels)
                break
            elif task_type == elasticdl_pb2.TRAINING:
                # TODO: optimize the logic to avoid unnecessary
                #       get_model call.
                if not train_with_local_model:
                    self.get_model()
                *accepted, min_model_version, loss = self._run_training_task(
                    features, labels
                )
                if accepted:
                    logger.info("Loss is {}".format(loss.numpy()))
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
        return min_model_version

    def _process_eval_task(self, task):
        """
        Check if there are evaluation tasks and process the tasks if any.
        Return:
            A python bool indicating whether worker processed some evaluation
            tasks.
        """
        logger.info("the evaluation task_id: %d" % task.task_id)
        eval_info = self._task_data_service.get_validation_dataset(task)
        if not eval_info:
            return
        (eval_dataset, model_version, task_id) = eval_info
        eval_dataset = self._dataset_fn(
            eval_dataset,
            Mode.EVALUATION,
            self._task_data_service.data_reader.metadata,
        )
        eval_dataset = eval_dataset.batch(self._minibatch_size).prefetch(1)
        err_msg = ""
        for dataset_batch in eval_dataset:
            data_err_msg = self._process_minibatch_and_report(
                dataset_batch, elasticdl_pb2.EVALUATION, model_version
            )
            if data_err_msg:
                err_msg = data_err_msg
                break
        del eval_dataset
        self.report_evaluation_metrics(
            model_outputs=self._evaluation_result[MetricsDictKey.MODEL_OUTPUT],
            labels=self._evaluation_result[MetricsDictKey.LABEL],
        )
        self.report_task_result(task_id, err_msg)
        self._evaluation_result = {}

    def _process_save_model_task_if_needed(self):
        (
            task,
            dataset,
        ) = self._task_data_service.get_save_model_task_and_dataset()
        if task is not None and dataset is not None:
            dataset = self._dataset_fn(
                dataset,
                Mode.PREDICTION,
                self._task_data_service.data_reader.metadata,
            )
            dataset = dataset.batch(self._minibatch_size)
            saved_model_path = task.extended_config.get(
                SaveModelConfig.SAVED_MODEL_PATH
            )
            saved_model_path = os.path.join(
                saved_model_path, str(int(time.time()))
            )
            logger.info(
                "The path to export model is {}".format(saved_model_path)
            )
            model = self._model_handler.get_model_to_export(
                self._model, dataset
            )
            tf.saved_model.save(model, saved_model_path)
            self.report_task_result(task_id=task.task_id, err_msg="")

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
                break
            dataset = self._dataset_fn(
                dataset,
                Mode.TRAINING,
                self._task_data_service.data_reader.metadata,
            )
            dataset = dataset.batch(self._minibatch_size).prefetch(1)
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
                self._task_data_service.report_record_done(
                    self._minibatch_size, err_msg
                )
            del dataset
            # New evaluation tasks may be created after this worker's
            # training tasks are done, as other workers' may still
            # have pending training tasks.
            if self._job_type == JobType.TRAINING_WITH_EVALUATION:
                evaluation_task_executed = self._evaluate_only()

            self._process_save_model_task_if_needed()

    def _evaluate_only(self):
        """
        Only evaluate the model on the worker.
        """
        evaluation_task_executed = False
        # get the latest model before processing eval tasks
        self.get_model()
        while True:
            task = self.get_task(elasticdl_pb2.EVALUATION)
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
