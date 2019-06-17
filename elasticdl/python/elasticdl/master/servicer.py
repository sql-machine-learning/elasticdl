import logging
import os
import threading
from collections import defaultdict

from google.protobuf import empty_pb2

from elasticdl.proto import elasticdl_pb2
from elasticdl.proto import elasticdl_pb2_grpc
from elasticdl.python.elasticdl.common.ndarray import (
    ndarray_to_tensor,
    tensor_to_ndarray,
)
from elasticdl.python.elasticdl.common.model_helper import (
    save_checkpoint_to_file,
    load_from_checkpoint_file,
)

import numpy as np

import tensorflow as tf

assert tf.executing_eagerly()


class MasterServicer(elasticdl_pb2_grpc.MasterServicer):
    """Master service implementation"""

    def __init__(
        self,
        grads_to_wait,
        minibatch_size,
        optimizer,
        task_q,
        *,
        init_var,
        init_from_checkpoint,
        checkpoint_dir,
        checkpoint_steps,
        keep_checkpoint_max
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
        self._evaluation_metrics = defaultdict(list)
        for var in init_var:
            self.set_model_var(var.name, var.numpy())
        self._var_created = len(init_var) > 0
        if init_from_checkpoint:
            self.load_checkpoint_file(init_from_checkpoint)
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_steps = checkpoint_steps
        self._keep_checkpoint_max = keep_checkpoint_max
        if not self._checkpoint_dir:
            self._checkpoint_dir = os.getcwd() + "/checkpoint_dir"
        if self._checkpoint_steps:
            os.makedirs(self._checkpoint_dir, exist_ok=True)
            if self._keep_checkpoint_max > 0:
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
            res.type = task.type
            # For evaluation task, it will use the fixed version model
            if task.type == elasticdl_pb2.EVALUATION:
                res.model_version = task.model_version
        return res

    def GetModel(self, request, _):
        self._validate_model_version(request.version)

        if (
            request.method == elasticdl_pb2.MINIMUM
            or request.version == self._version
        ):
            with self._lock:
                res = self._get_model_no_lock()
            return res

        # Read from checkpoint for the fixed version model
        pb_model = elasticdl_pb2.Model()
        try:
            file_name = self._get_checkpoint_file_path(request.version)
            pb_model = load_from_checkpoint_file(file_name)
        except Exception:
            self._logger.error(
                "Failed to fetch checkpoint model for "
                "model version {}".format(request.version)
            )
        return pb_model

    def _update_model_version(self):
        assert self._lock.locked()
        self._version += 1

    def _get_checkpoint_file_path(self, version):
        return "{}/model_v{}.chkpt".format(self._checkpoint_dir, version)

    def _get_version_from_checkpoint_file(self, file_path):
        # TODO: we need to move checkpoint-related code to a separate file,
        #       and each checkpoint object should be a version + file pair.
        #       Thus, we don't need to parse from file name.
        file_name = os.path.basename(file_path)
        return int(file_name.split(".")[0][7:])

    def _create_var_from_tensor_dict(self, tensor_dict):
        for k, v in tensor_dict.items():
            self.set_model_var(k, tensor_to_ndarray(v))
        if tensor_dict:
            self._var_initialized = True

    def save_checkpoint(self):
        file_name = self._get_checkpoint_file_path(self._version)
        pb_model = self._get_model_no_lock()
        save_checkpoint_to_file(pb_model, file_name)
        if self._keep_checkpoint_max:
            self._checkpoint_list.append(file_name)
            while len(self._checkpoint_list) > self._keep_checkpoint_max:
                file_to_delete = self._checkpoint_list.pop(0)
                os.remove(file_to_delete)

    def load_checkpoint_file(self, file_name):
        pb_model = load_from_checkpoint_file(file_name)

        if self._var_created:
            for k, v in self._model.items():
                # Assumes all variables exist in pb_model.param.
                v.assign(tensor_to_ndarray(pb_model.param[k]))
        else:
            # Create variables from pb_model.param
            self._create_var_from_tensor_dict(pb_model.param)
        self._version = pb_model.version

    def get_last_checkpoint_version(self):
        if not self._checkpoint_list:
            raise RuntimeError("No model checkpoint available")
        last_checkpoint_file = self._checkpoint_list[-1]
        return self._get_version_from_checkpoint_file(last_checkpoint_file)

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
            self._logger.warning(err_msg)
            raise ValueError(err_msg)
        return request_model_version == self._version

    def ReportVariable(self, request, _):
        res = elasticdl_pb2.ReportVariableResponse()
        with self._lock:
            if not self._var_created:
                self._create_var_from_tensor_dict(request.variable)
                self._var_created= True
        res.var_created = self._var_created
        return res

    def ReportGradient(self, request, _):
        model_version_valid = self._validate_model_version(
            request.model_version
        )

        res = elasticdl_pb2.ReportGradientResponse()
        if not model_version_valid:
            self._logger.warning(
                "Task result for outdated version %d dropped",
                request.model_version,
            )
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
                if self._checkpoint_steps and (
                    self._version % self._checkpoint_steps == 0
                ):
                    try:
                        self.save_checkpoint()
                    except Exception:
                        self._logger.warning(
                            "Failed to save checkpoint file for "
                            "model version {}".format(self._version)
                        )

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
        model_version_valid = self._validate_model_version(
            request.model_version
        )

        res = elasticdl_pb2.ReportEvaluationMetricsResponse()
        if not model_version_valid:
            res.accepted = False
            res.model_version = self._version
            return res

        with self._lock:
            for k, v in request.evaluation_metrics.items():
                if v.dim:
                    self._evaluation_metrics[k].append(tensor_to_ndarray(v))
            evaluation_metrics_summary = {
                k: np.mean(v) for k, v in self._evaluation_metrics.items()
            }
            self._logger.info(
                "Evaluation metrics: %s" % evaluation_metrics_summary
            )
        res.accepted = True
        res.model_version = self._version
        return res
