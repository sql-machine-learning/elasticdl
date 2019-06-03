import logging
import os
import threading
import numpy as np

import tensorflow as tf

assert tf.executing_eagerly()

from google.protobuf import empty_pb2

from elasticdl.proto import elasticdl_pb2
from elasticdl.proto import elasticdl_pb2_grpc
from elasticdl.python.elasticdl.common.ndarray import ndarray_to_tensor, tensor_to_ndarray
from elasticdl.python.elasticdl.common.model_helper import save_checkpoint_to_file, load_from_checkpoint_file


class MasterServicer(elasticdl_pb2_grpc.MasterServicer):
    """Master service implementation"""

    def __init__(
        self,
        grads_to_wait,
        minibatch_size,
        optimizer,
        task_q,
        *,
        init_var=[],
        checkpoint_dir="",
        save_checkpoint_steps=0,
        keep_checkpoint_max=3
    ):
        # TODO: group params together into a single object.
        self._logger = logging.getLogger(__name__)
        self._opt = optimizer
        self._task_q = task_q
        self._lock = threading.Lock()
        # A <string, tf.ResourceVariable> map. We use tf.ResourceVariable
        # instead ndarray to avoid copying and conversion when calling
        # optimizer's apply_gradients() function.
        self._model = {}
        self._version = 0
        self._gradient_sum = {}
        self._grad_to_wait = grads_to_wait
        self._grad_n = 0
        self._minibatch_size = minibatch_size
        self._evaluation_metrics = {}
        for var in init_var:
            self.set_model_var(var.name, var.numpy())
        self._checkpoint_dir = checkpoint_dir
        self._save_checkpoint_steps = save_checkpoint_steps
        self._keep_checkpoint_max = keep_checkpoint_max
        if self._save_checkpoint_steps and not self._checkpoint_dir:
            self.logger.warning(
                "checkpoint_dir not set, checkpint files will be saved in %s",
                os.getcwd()
            )
            self._checkpoint_dir = os.getcwd()
        if self._save_checkpoint_steps and self._keep_checkpoint_max:
            self._checkpoint_list = []

    def set_model_var(self, name, value):
        """Add or set model variable. Value should be a float32 ndarray"""
        if value.dtype != np.float32:
            raise ValueError("Value should be a float32 numpy array")
        self._model[name] = tf.Variable(
            value, name=MasterServicer.var_name_encode(name)
        )

    @staticmethod
    def var_name_encode(name):
        return name.replace(":", "-")

    def GetTask(self, request, _):
        res = elasticdl_pb2.Task()
        res.model_version = self._version
        res.minibatch_size = self._minibatch_size
        task_id, task = self._task_q.get(request.worker_id)
        if task:
            res.task_id = task_id
            res.shard_file_name = task.file_name
            res.start = task.start
            res.end = task.end
        return res

    def GetModel(self, request, _):
        _ = self._validate_model_version(request.min_version)

        res = elasticdl_pb2.Model()
        with self._lock:
            res = self._get_model_no_lock(res)
        return res

    def _update_model_version(self):
        assert self._lock.locked()
        self._version += 1

    def save_checkpoint(self):
        file_name = "%s/model_v%d.chkpt".format(self._checkpoint_dir, self._version)
        pb_model = elasticdl_pb2.Model()
        pb_model = self._get_model_no_lock(pb_model)
        save_checkpoint_to_file(pb_model, file_name)

    def load_checkpoint_file(self, file_name):
        pb_model = elasticdl_pb2.Model()
        pb_model = load_from_checkpoint_file(pb_model, file_name)

        for k, v in self._model.items():
            # Assumes all variables exist in pb_model.param.
            v.assign(
                tensor_to_ndarray(pb_model.param[k]))
        self._model_version = pb_model.version

    def _update_model(self):
        assert self._lock.locked()
        grad_var = []
        for k in self._gradient_sum:
            self._gradient_sum[k] = self._gradient_sum[k] / self._grad_to_wait
            grad_var.append((self._gradient_sum[k], self._model[k]))
        self._opt.apply_gradients(grad_var)
        self._update_model_version()
        self._gradient_sum.clear()
        self._grad_n = 0

    def _get_model_no_lock(self, pb_model):
        pb_model.version = self._version
        for k, v in self._model.items():
            pb_model.param[k].CopyFrom(ndarray_to_tensor(v.numpy()))
        return pb_model

    def _validate_model_version(self, request_model_version):
        if request_model_version > self._version:
            err_msg = "Model version %d not available yet, current version: %d" % (
                request_model_version,
                self._version,
            )
            self._logger.warning(err_msg)
            raise ValueError(err_msg)

        invalid_model_version = request_model_version < self._version
        if invalid_model_version:
            self._logger.warning(
                "Task result for outdated version %d dropped",
                request_model_version,
            )
        return invalid_model_version

    def ReportGradient(self, request, _):
        invalid_model_version = self._validate_model_version(request.model_version)

        res = elasticdl_pb2.ReportGradientResponse()
        if invalid_model_version:
            res.accepted = False
            res.model_version = self._version
            return res

        # TODO: Update task queue with task_id
        with self._lock:
            tmp = {}
            # Do sanity check before accumulating gradients.
            for k, v in request.gradient.items():
                if k not in self._model:
                    raise ValueError(
                        "Gradient key: %s is not part of model", k
                    )
                arr = tensor_to_ndarray(v)
                if arr.shape != self._model[k].numpy().shape:
                    raise ValueError(
                        "Gradient key: %s has incompatible dimension", k
                    )
                tmp[k] = arr

            for k, v in tmp.items():
                if k in self._gradient_sum:
                    self._gradient_sum[k] = self._gradient_sum[k] + v
                else:
                    self._gradient_sum[k] = v

            self._grad_n += 1
            if self._grad_n >= self._grad_to_wait:
                self._update_model()
                if self._save_checkpoint_steps and self._version % self._save_checkpoint_steps == 0:
                    self.save_checkpoint()

        res.accepted = True
        res.model_version = self._version
        return res

    def ReportTaskResult(self, request, _):
        if request.err_message:
            self._logger.warning(
                "Worker reported error: " + request.err_message
            )
            self._task_q.report(request.task_id, False)
        else:
            self._task_q.report(request.task_id, True)
        return empty_pb2.Empty()

    def ReportEvaluationMetrics(self, request, _):
        invalid_model_version = self._validate_model_version(request.model_version)

        res = elasticdl_pb2.ReportEvaluationMetricsResponse()
        if invalid_model_version:
            res.accepted = False
            res.model_version = self._version
            return res

        with self._lock:
            for k, v in request.evaluation_metrics.items():
                arr = tensor_to_ndarray(v)
                self._evaluation_metrics[k] = arr

            self._update_model_version()
        res.accepted = True
        res.model_version = self._version
        return res
