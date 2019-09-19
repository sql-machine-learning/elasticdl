import threading

import numpy as np
import tensorflow as tf
from google.protobuf import empty_pb2

from elasticdl.proto import elasticdl_pb2, elasticdl_pb2_grpc
from elasticdl.python.common.file_helper import copy_if_not_exists
from elasticdl.python.common.log_util import default_logger as logger
from elasticdl.python.common.model_helper import load_from_checkpoint_file
from elasticdl.python.common.ndarray import (
    ndarray_to_tensor,
    tensor_to_ndarray,
)
from elasticdl.python.common.tensor_helper import merge_indexed_slices
from elasticdl.python.elasticdl.layers.embedding import Embedding
from elasticdl.python.master.checkpoint_service import CheckpointService
from elasticdl.python.master.lr_modulation import (
    add_lr_modulation_to_optimizer,
)
from elasticdl.python.master.optimizer_wrapper import OptimizerWrapper


class MasterServicer(elasticdl_pb2_grpc.MasterServicer):
    """Master service implementation"""

    def __init__(
        self,
        grads_to_wait,
        minibatch_size,
        optimizer,
        task_d,
        *,
        init_var,
        checkpoint_filename_for_init,
        checkpoint_service,
        evaluation_service,
        embedding_service=None,
        embedding_dims={},
        lr_staleness_modulation=False,
        use_async=False,
    ):
        # TODO: group params together into a single object.
        self._task_d = task_d
        self._lock = threading.Lock()
        self._gradient_sum = {}
        self._edl_embedding_gradients = {}
        self._gradient_sum_indexed = {}
        self._grad_to_wait = grads_to_wait
        self._grad_n = 0
        self._minibatch_size = minibatch_size
        self._use_async = use_async
        self._lr_staleness_modulation = lr_staleness_modulation

        # A <string, tf.ResourceVariable> map. We use tf.ResourceVariable
        # instead ndarray to avoid copying and conversion when calling
        # optimizer's apply_gradients() function.
        self._model = {}
        self._version = 0
        self._embedding_service = embedding_service
        self._init_model(checkpoint_filename_for_init, init_var)
        self._opt = self._init_optimizer(
            optimizer, embedding_service, embedding_dims, use_async
        )

        self._checkpoint_service = checkpoint_service
        self._evaluation_service = evaluation_service
        if evaluation_service:
            evaluation_service.set_master_servicer(self)

    # TODO: Multiple tests are currently using the function `set_model_var` to
    # initialize self._model, where the initialization should be done via
    # servicer's constructor.
    def set_model_var(self, name, value):
        """Add or set model variable. Value should be a float32 ndarray"""
        if value.dtype != np.float32:
            raise ValueError("Value should be a float32 numpy array")
        self._model[name] = tf.Variable(
            value, name=MasterServicer.var_name_encode(name)
        )

    def _modulate_lr_if_needed(self, opt):
        if self._use_async and self._lr_staleness_modulation:
            self._lr_modulation = add_lr_modulation_to_optimizer(opt)
        else:
            self._lr_modulation = None

    def _init_model_from_var_list(self, var_list):
        for var in var_list:
            self.set_model_var(var.name, var.numpy())

    def _init_model_from_tensor_dict(self, tensor_dict):
        assert tensor_dict
        for name, val in tensor_dict.items():
            self.set_model_var(name, tensor_to_ndarray(val))

    def _init_model(self, checkpoint_filename_for_init, init_var):
        if checkpoint_filename_for_init:
            pb_model = load_from_checkpoint_file(checkpoint_filename_for_init)
            self._version = pb_model.version
            self._init_model_from_tensor_dict(pb_model.param)
        elif init_var:
            self._init_model_from_var_list(init_var)
        else:
            logger.info(
                "Model is not intialized. It will be "
                "initialized by the first update from "
                "the worker."
            )

    def _init_optimizer(
        self, opt, embedding_service, embedding_dims, use_async
    ):
        # `embedding_service` is not None means ElasticDL embedding
        # layers are used
        self._modulate_lr_if_needed(opt)
        if embedding_service:
            return OptimizerWrapper(
                opt, embedding_service, embedding_dims, use_async
            )
        return opt

    @staticmethod
    def var_name_encode(name):
        return name.replace(":", "-")

    def GetTask(self, request, _):
        res = elasticdl_pb2.Task()
        res.model_version = self._version
        res.minibatch_size = self._minibatch_size
        task_id, task = self._task_d.get(request.worker_id)
        if task:
            res.task_id = task_id
            res.shard_name = task.shard_name
            res.start = task.start
            res.end = task.end
            res.type = task.type
            # For evaluation task, it will use the fixed version model
            if task.type == elasticdl_pb2.EVALUATION:
                res.model_version = task.model_version
        elif not self._task_d.finished():
            # Not all tasks are finished, wait in case of new tasks later.
            res.type = elasticdl_pb2.WAIT
        return res

    def GetModel(self, request, _):
        if not self._use_async:
            self._validate_model_version(request.version)

        if (
            request.method == elasticdl_pb2.MINIMUM
            or request.version == self._version
        ):
            if self._use_async:
                res = self._get_model_no_lock()
            else:
                with self._lock:
                    res = self._get_model_no_lock()
            return res

        # Read from checkpoint for the fixed version model
        pb_model = elasticdl_pb2.Model()
        try:
            pb_model = self._checkpoint_service.get_checkpoint_model(
                request.version
            )
        except Exception:
            logger.error(
                "Failed to fetch checkpoint model for "
                "model version {}".format(request.version)
            )
        return pb_model

    def _update_model_version(self):
        assert self._lock.locked()
        self._version += 1

    def _update_edl_embedding_table(self, name_var_list):
        """
            Put updated embedding vectors' ids and values together
            and use EmbeddingService.update_embedding() to update
            embedding table in the distributed storage
        """
        keys = []
        embeddings = []
        for layer_name, unique_ids, embedding_var in name_var_list:
            keys.extend(
                [
                    Embedding.get_key([layer_name, i])
                    for i in unique_ids.numpy()
                ]
            )
            embeddings.extend([i for i in embedding_var.numpy()])

        if embeddings:
            self._embedding_service.update_embedding(
                key=keys, embedding_vectors=embeddings
            )

    def _update_model(self):
        grad_var = []

        # (grad, var) pairs excluding keras Embedding layer and
        # ElasticDL Embedding layer
        for k in self._gradient_sum:
            if not self._use_async:
                self._gradient_sum[k] = (
                    self._gradient_sum[k] / self._grad_to_wait
                )
            grad_var.append((self._gradient_sum[k], self._model[k]))

        # (grad, var) pair of Keras Embedding layer
        for k in self._gradient_sum_indexed:
            grad_var.append((self._gradient_sum_indexed[k], self._model[k]))

        # (grad, var) pair of ElasticDL Embedding layer
        if self._edl_embedding_gradients:
            for layer_name, grads in self._edl_embedding_gradients.items():
                grad_var.append((grads, layer_name))

        self._opt.apply_gradients(grad_var)

        # need the lock for model version update in async SGD
        if self._use_async:
            self._lock.acquire()
        self._update_model_version()
        self._update_evaluation()
        self._update_checkpoint()
        if self._use_async:
            self._lock.release()
        else:
            self._gradient_sum.clear()
            self._gradient_sum_indexed.clear()
            self._edl_embedding_gradients.clear()
            self._grad_n = 0

    def get_model_version(self):
        return self._version

    def _save_checkpoint(self, locking, is_eval_checkpoint):
        try:
            logger.info(
                "Saving checkpoint for model version %d" % self._version
            )
            if locking:
                self._lock.acquire()
            pb_model = self._get_model_no_lock()
            self._checkpoint_service.save(
                self._version, pb_model, is_eval_checkpoint
            )
            checkpoint_version = self._version
            if locking:
                self._lock.release()
            return checkpoint_version
        except Exception:
            logger.error(
                "Failed to save checkpoint file for model version %d"
                % self._version
            )

    def save_latest_checkpoint(self, output_path):
        if self._checkpoint_service is None:
            self._checkpoint_service = CheckpointService(
                checkpoint_dir="",
                checkpoint_steps=1,
                keep_checkpoint_max=1,
                include_evaluation=False,
            )
        self._save_checkpoint(locking=False, is_eval_checkpoint=False)
        checkpoint_path = self._checkpoint_service.get_checkpoint_path(
            self._checkpoint_service.get_latest_checkpoint_version()
        )
        copy_if_not_exists(checkpoint_path, output_path, is_dir=False)

    def _update_evaluation(self):
        if self._evaluation_service:
            self._evaluation_service.add_evaluation_task_if_needed(
                master_locking=False
            )

    def _update_checkpoint(self):
        if (
            self._checkpoint_service
            and self._checkpoint_service.need_to_checkpoint(self._version)
        ):
            self._save_checkpoint(locking=False, is_eval_checkpoint=False)

    def _get_model_no_lock(self):
        pb_model = elasticdl_pb2.Model()
        pb_model.version = self._version
        for k, v in self._model.items():
            pb_model.param[k].CopyFrom(ndarray_to_tensor(v.numpy()))
        return pb_model

    def _validate_model_version(self, request_model_version):
        if request_model_version > self._version:
            err_msg = (
                "Model version %d not available yet, "
                "current version: %d" % (request_model_version, self._version)
            )
            logger.warning(err_msg)
            raise ValueError(err_msg)
        return request_model_version == self._version

    def ReportVariable(self, request, _):
        with self._lock:
            if not self._model:
                self._init_model_from_tensor_dict(request.variable)
        return empty_pb2.Empty()

    def ReportGradient(self, request, _):
        model_version_valid = self._use_async or self._validate_model_version(
            request.model_version
        )

        res = elasticdl_pb2.ReportGradientResponse()
        if not model_version_valid:
            logger.warning(
                "Task result for outdated version %d dropped",
                request.model_version,
            )
            res.accepted = False
            res.model_version = self._version
            return res

        tmp = {}
        indexed_grads = {}
        edl_embedding_gradients = {}
        # Do sanity check before accumulating gradients.
        for k, v in request.gradient.items():
            if k not in self._model:
                if v.indices:
                    # grads of ElasticDL Embedding layer
                    # TODO: check arr.shape[1] = embedding_dim of this
                    # EdlEmbedding layer
                    arr = tensor_to_ndarray(v)
                    edl_embedding_gradients[k] = arr
                    continue
                else:
                    raise ValueError(
                        "Gradient key: %s is not part of model", k
                    )

            arr = tensor_to_ndarray(v)
            if isinstance(arr, tf.IndexedSlices):
                if arr.values.shape[1] != self._model[k].numpy().shape[1]:
                    raise ValueError(
                        "Gradient key: %s has incompatible "
                        "indexed slice dimension %d, expected %d"
                        % (
                            k,
                            arr.values.shape[1],
                            self._model[k].numpy().shape[1],
                        )
                    )

                max_index = tf.math.reduce_max(arr.indices).numpy()
                if max_index >= self._model[k].numpy().shape[0]:
                    raise ValueError(
                        "Gradient key: %s has wrong indices %d, "
                        "out of range %d"
                        % (k, max_index, self._model[k].numpy().shape[0] - 1)
                    )
                indexed_grads[k] = arr
            else:
                if arr.shape != self._model[k].numpy().shape:
                    raise ValueError(
                        "Gradient key: %s has incompatible dimension", k
                    )
                tmp[k] = arr

        if not self._use_async:
            self._lock.acquire()
        self._process_gradients(
            edl_embedding_gradients, indexed_grads, tmp, request.model_version
        )
        if not self._use_async:
            self._lock.release()

        res.accepted = True
        res.model_version = self._version
        return res

    def _process_gradients(
        self, edl_embedding_gradients, indexed_grads, grads, request_version
    ):
        if not self._use_async:
            # grads of ElasticDL Embedding layer
            for k, v in edl_embedding_gradients.items():
                if k in self._edl_embedding_gradients:
                    self._edl_embedding_gradients[k] = merge_indexed_slices(
                        self._edl_embedding_gradients[k], v
                    )
                else:
                    self._edl_embedding_gradients[k] = v

            # grads of Keras Embedding layer
            for k, v in indexed_grads.items():
                if k not in self._gradient_sum_indexed:
                    self._gradient_sum_indexed[k] = v
                else:
                    grads_s = self._gradient_sum_indexed[k]
                    self._gradient_sum_indexed[k] = merge_indexed_slices(
                        grads_s, v
                    )

            # other grads
            for k, v in grads.items():
                if not self._use_async and k in self._gradient_sum:
                    self._gradient_sum[k] = self._gradient_sum[k] + v
                else:
                    self._gradient_sum[k] = v
            self._grad_n += 1
        else:
            # TODO: do not accumulate gradients but apply directly.
            pass

        # staleness-aware learning rate modulation
        if self._lr_modulation:
            staleness = max(1, self._version - request_version)
            self._lr_modulation.set_multiplier(1.0 / staleness)
        if self._use_async or self._grad_n >= self._grad_to_wait:
            self._update_model()

    def ReportTaskResult(self, request, _):
        if request.err_message:
            logger.warning("Worker reported error: " + request.err_message)
            self._task_d.report(request.task_id, False)
        else:
            self._task_d.report(request.task_id, True)
        return empty_pb2.Empty()

    def ReportEvaluationMetrics(self, request, _):
        report_metrics = self._evaluation_service.report_evaluation_metrics(
            request.model_version, request.evaluation_metrics
        )
        res = elasticdl_pb2.ReportEvaluationMetricsResponse()
        res.model_version = self._version
        res.accepted = report_metrics
        return res
