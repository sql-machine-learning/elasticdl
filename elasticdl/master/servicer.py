import threading
import numpy as np

import tensorflow as tf
from proto import master_pb2
from proto import master_pb2_grpc
from util.ndarray import ndarray_to_tensor, tensor_to_ndarray


class MasterServicer(master_pb2_grpc.MasterServicer):
    """Master service implementation"""

    def __init__(self, logger, grads_to_wait):
        self.logger = logger
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

    def _set_model_var(self, name, value):
        """Add or set model variable. Value should be a float32 ndarray"""
        if value.dtype != np.float32:
            raise ValueError("Value should be a float32 numpy array")
        self._model[name] = tf.Variable(value, name=name, use_resource=True)

    def GetTask(self, request, context):
        # TODO: implent task queues. Return an empty task for now.
        res = master_pb2.Task()
        res.shard_file_name = ""
        res.model_version = self._version
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

    def ReportTaskResult(self, request, context):
        if request.model_version > self._version:
            err_msg = "Model version %d out of range, current version: %d" % (
                request.model_version,
                self._version,
            )
            self.logger.warning(err_msg)
            raise ValueError(err_msg)

        res = master_pb2.ReportTaskResultReply()
        if request.model_version < self._version:
            self.logger.warning(
                "Task result for outdated version %d dropped",
                request.model_version,
            )
            res.accepted = False
            res.model_version = self._version
            return res

        if request.err_message:
            self.logger.warning("Worker error: %s" % request.err_message)
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
                # TODO: update model
                self._version += 1
                self._gradient_sum.clear()
                self._grad_n = 0
        res.accepted = True
        res.model_version = self._version
        return res
