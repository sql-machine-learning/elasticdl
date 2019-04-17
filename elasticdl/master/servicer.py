import threading
import numpy as np

import tensorflow as tf
assert tf.executing_eagerly()

from google.protobuf import empty_pb2

from proto import master_pb2
from proto import master_pb2_grpc
from common.ndarray import ndarray_to_tensor, tensor_to_ndarray


class MasterServicer(master_pb2_grpc.MasterServicer):
    """Master service implementation"""

    def __init__(self, logger, grads_to_wait, minibatch_size, optimizer, task_q):
        # TODO: group params together into a single object.
        self.logger = logger
        self._opt = optimizer
        self._task_q = task_q
        self._lock = threading.Lock()
        # TODO: random initialization
        # A <string, tf.ResourceVariable> map. We use tf.ResourceVariable
        # instead ndarray to avoid copying and conversion when calling
        # optimizer's apply_gradients() function.
        self._model = {}
        self._version = 0
        self._gradient_sum = {}
        self._grad_to_wait = grads_to_wait
        self._grad_n = 0
        self._minibatch_size = minibatch_size

    def set_model_var(self, name, value):
        """Add or set model variable. Value should be a float32 ndarray"""
        if value.dtype != np.float32:
            raise ValueError("Value should be a float32 numpy array")
        self._model[name] = tf.Variable(value, name=MasterServicer.var_name_encode(name))


    @staticmethod
    def var_name_encode(name):
        return name.replace(":", "-")


    def GetTask(self, request, context):
        res = master_pb2.Task()
        res.model_version = self._version
        res.minibatch_size = self._minibatch_size
        task_id, task = self._task_q.get()
        if task:
            res.task_id = task_id
            res.shard_file_name = task.file_name
            res.start = task.start
            res.end = task.end
        return res


    def GetModel(self, request, context):
        if request.min_version > self._version:
            err_msg = (
                "Requested version %d not available yet, current version: %d"
                % (request.min_version, self._version)
            )
            self.logger.warning(err_msg)
            raise ValueError(err_msg)

        res = master_pb2.Model()
        with self._lock:
            res.version = self._version
            for k, v in self._model.items():
                res.param[k].CopyFrom(ndarray_to_tensor(v.numpy()))
        return res

    def _update_model(self):
        assert self._lock.locked()
        grad_var = []
        for k in self._gradient_sum:
            self._gradient_sum[k] = self._gradient_sum[k] / self._grad_to_wait
            grad_var.append((self._gradient_sum[k], self._model[k]))
        self._opt.apply_gradients(grad_var)
        self._version += 1
        self._gradient_sum.clear()
        self._grad_n = 0

    def ReportGradient(self, request, context):
        if request.model_version > self._version:
            err_msg = "Model version %d out of range, current version: %d" % (
                request.model_version,
                self._version,
            )
            self.logger.warning(err_msg)
            raise ValueError(err_msg)

        res = master_pb2.ReportGradientReply()
        if request.model_version < self._version:
            self.logger.warning(
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
        res.accepted = True
        res.model_version = self._version
        return res

    def ReportTaskResult(self, request, context):
        if request.err_message:
            self.logger.warning("Worker reported error: " + request.err_message)
            self._task_q.report(request.task_id, False)
        else:
            self._task_q.report(request.task_id, True)
        return empty_pb2.Empty()
