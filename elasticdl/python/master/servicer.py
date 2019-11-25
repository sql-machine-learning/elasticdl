import threading

import numpy as np
import tensorflow as tf
from google.protobuf import empty_pb2

from elasticdl.proto import elasticdl_pb2, elasticdl_pb2_grpc
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.common.model_utils import load_from_checkpoint_file
from elasticdl.python.common.tensor import tensor_pb_to_ndarray
from elasticdl.python.master.learning_rate_modulator import (
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
        # instead of ndarray to avoid copying and conversion when calling
        # optimizer's apply_gradients() function.
        self._model = {}
        self._version = 0
        self._init_model(checkpoint_filename_for_init, init_var)
        self._opt = self._init_optimizer(optimizer, use_async)

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

    def _init_model_from_tensor_pb_list(self, tensor_pb_list):
        assert tensor_pb_list
        for pb in tensor_pb_list:
            self.set_model_var(pb.name, tensor_pb_to_ndarray(pb))

    def _init_model(self, checkpoint_filename_for_init, init_var):
        if checkpoint_filename_for_init:
            pb_model = load_from_checkpoint_file(checkpoint_filename_for_init)
            self._version = pb_model.version
            self._init_model_from_tensor_pb_list(pb_model.param)
        elif init_var:
            self._init_model_from_var_list(init_var)
        else:
            logger.info(
                "Model is not intialized. It will be "
                "initialized by the first update from "
                "the worker."
            )

    def _init_optimizer(self, opt, use_async):
        self._modulate_lr_if_needed(opt)
        return OptimizerWrapper(opt, None, None, use_async)

    @staticmethod
    def var_name_encode(name):
        return name.replace(":", "-")

    def GetTask(self, request, _):
        res = elasticdl_pb2.Task()
        res.model_version = self._version
        res.minibatch_size = self._minibatch_size
        if request.task_type == elasticdl_pb2.EVALUATION:
            task_id, task = self._task_d.get_eval_task(request.worker_id)
        else:
            task_id, task = self._task_d.get(request.worker_id)

        if task:
            res.task_id = task_id
            res.shard_name = task.shard_name
            res.start = task.start
            res.end = task.end
            res.type = task.type
            for k, v in task.extended_config.items():
                res.extended_config[k] = v

            # For evaluation task, it will use the fixed version model
            if task.type == elasticdl_pb2.EVALUATION:
                res.model_version = task.model_version
        elif (not self._task_d.finished()) or (
            self._task_d.invoke_deferred_callback()
        ):
            # If the todo and doing tasks are not empty,
            # Otherwise if the callback list is not empty,
            # we are trying to pop and invoke the callback.
            # Then the master tells the worker to wait
            # in case of new tasks later.
            res.type = elasticdl_pb2.WAIT

        return res

    def _update_evaluation(self):
        if self._evaluation_service:
            self._evaluation_service.add_evaluation_task_if_needed(
                master_locking=False, model_version=self._version
            )

    def ReportTaskResult(self, request, _):
        if request.err_message:
            logger.warning("Worker reported error: " + request.err_message)
            self._task_d.report(request, False)
        else:
            self._task_d.report(request, True)
        return empty_pb2.Empty()

    def ReportEvaluationMetrics(self, request, _):
        report_metrics = self._evaluation_service.report_evaluation_metrics(
            request.model_version, request.model_outputs, request.labels
        )
        res = elasticdl_pb2.ReportEvaluationMetricsResponse()
        res.model_version = self._version
        res.accepted = report_metrics
        return res

    def ReportVersion(self, request, _):
        if self._evaluation_service:
            self._evaluation_service.add_evaluation_task_if_needed(
                master_locking=False, model_version=request.model_version
            )
        return empty_pb2.Empty()
