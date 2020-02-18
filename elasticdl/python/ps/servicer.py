import threading

import tensorflow as tf
from google.protobuf import empty_pb2
from tensorflow.core.framework import tensor_pb2

import elasticdl.python.common.tensor_utils as tensor_utils
from elasticdl.proto import elasticdl_pb2, elasticdl_pb2_grpc
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.common.tensor_utils import (
    merge_indexed_slices,
    ndarray_to_pb,
    pb_to_indexed_slices,
    pb_to_ndarray,
    serialize_ndarray,
)
from elasticdl.python.ps.optimizer_wrapper import OptimizerWrapper


class PserverServicer(elasticdl_pb2_grpc.PserverServicer):
    """PS service implementation"""

    def __init__(
        self,
        parameters,
        grads_to_wait,
        optimizer,
        lr_scheduler="",
        lr_staleness_modulation=False,
        sync_version_tolerance=0,
        use_async=False,
        evaluation_steps=0,
        master_channel=None,
        checkpoint_saver=None,
        ps_id=None,
        num_ps_pods=None,
    ):
        if master_channel is None:
            self._master_stub = None
        else:
            self._master_stub = elasticdl_pb2_grpc.MasterStub(master_channel)

        self._parameters = parameters
        self._grads_to_wait = grads_to_wait
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._lr_staleness_modulation = lr_staleness_modulation
        self._sync_version_tolerance = sync_version_tolerance
        self._use_async = use_async
        self._eval_steps = evaluation_steps
        self._checkpoint_saver = checkpoint_saver
        self._ps_id = ps_id
        self._num_ps_pods = num_ps_pods
        self._version_lock = threading.Lock()
        self._lock = threading.Lock()
        self._use_wrap_opt = False

        self._grads_n = 0
        self._grads_buffer = {}

    def pull_dense_parameters(self, request, _):
        """
        Response with all non-embedding parameters if initialized.
        """
        res = elasticdl_pb2.PullDenseParametersResponse()
        if not self._parameters.init_status:
            res.initialized = False
            return res

        # Only sync-SGD needs lock
        # TODO: use a read-write lock to support multiple concurrent reads
        if not self._use_async:
            self._lock.acquire()
        res.version = self._parameters.version
        # No need to send variables if the requester has the latest version.
        if self._parameters.version > request.version:
            for name, var in self._parameters.non_embedding_params.items():
                res.dense_parameters[name].CopyFrom(ndarray_to_pb(var.numpy()))
        if not self._use_async:
            self._lock.release()
        res.initialized = True
        return res

    def pull_embedding_vectors(self, request, _):
        result = tensor_pb2.TensorProto()
        if not request.ids:
            return result
        embedding_vectors = self._parameters.get_embedding_param(
            request.name, request.ids
        )
        serialize_ndarray(embedding_vectors, result)
        return result

    def push_model(self, request, _):
        with self._lock:
            accepted = self._parameters.init_from_model_pb(request)
        if accepted and self._parameters.has_embedding_params():
            self.wrap_optimizer_and_set_slot()
        return empty_pb2.Empty()

    def push_embedding_info(self, request, _):
        with self._lock:
            self._parameters.init_embedding_params(
                request.embedding_table_info
            )
            self.wrap_optimizer_and_set_slot()
        return empty_pb2.Empty()

    def push_gradients(self, request, _):
        res = elasticdl_pb2.PushGradientsResponse()
        if self._use_async:
            grad_vars = []

            for name, pb in request.dense_parameters.items():
                grad = pb_to_ndarray(pb)
                self._parameters.check_grad(
                    tensor_utils.Tensor(name, grad, None)
                )
                grad = tf.constant(grad)
                var = self._parameters.get_non_embedding_param(name)
                grad_vars.append((grad, var))

            for name, pb in request.embedding_tables.items():
                grad = pb_to_indexed_slices(pb)
                self._parameters.check_grad(
                    tensor_utils.Tensor(name, grad.values, grad.indices)
                )
                if name in self._parameters.non_embedding_params:
                    var = self._parameters.get_non_embedding_param(name)
                    grad_vars.append((grad, var))
                else:
                    grad_vars.append((grad, name))
            if self._lr_scheduler:
                self._lr_scheduler.set_model_version(self._parameters.version)
            self._optimizer.apply_gradients(grad_vars)
            with self._version_lock:
                self._parameters.version += 1
                self._save_params_to_checkpoint_if_needed()
                version = self._parameters.version
            self._report_version_if_needed(version)

            res.accepted = True
            res.version = self._parameters.version
            return res
        else:
            if (
                request.version
                < self._parameters.version - self._sync_version_tolerance
            ):
                res.accepted = False
                res.version = self._parameters.version
                return res

            with self._lock:
                for name, pb in request.dense_parameters.items():
                    grad = pb_to_ndarray(pb)
                    self._parameters.check_grad(
                        tensor_utils.Tensor(name, grad, None)
                    )
                    if name in self._grads_buffer:
                        self._grads_buffer[name] = (
                            self._grads_buffer[name] + grad
                        )
                    else:
                        self._grads_buffer[name] = grad

                for name, pb in request.embedding_tables.items():
                    grad = pb_to_indexed_slices(pb)
                    self._parameters.check_grad(
                        tensor_utils.Tensor(name, grad.values, grad.indices)
                    )
                    if name in self._grads_buffer:
                        self._grads_buffer[name] = merge_indexed_slices(
                            self._grads_buffer[name], grad
                        )
                    else:
                        self._grads_buffer[name] = grad

                self._grads_n += 1
                res.accepted = True

                updated_version = False
                version = self._parameters.version
                if self._grads_n == self._grads_to_wait:
                    grad_vars = []
                    for name, grad in self._grads_buffer.items():
                        # Dense gradients are averaged,
                        # while sparse gradients are summed
                        if not isinstance(grad, tf.IndexedSlices):
                            grad = grad / self._grads_to_wait
                            grad = tf.constant(grad)
                        var = self._parameters.get_non_embedding_param(name)
                        if var is None:
                            grad_vars.append((grad, name))
                        else:
                            grad_vars.append((grad, var))

                    if self._lr_scheduler:
                        self._lr_scheduler.set_model_version(
                            self._parameters.version
                        )
                    self._optimizer.apply_gradients(grad_vars)
                    self._grads_n = 0
                    self._grads_buffer.clear()
                    self._parameters.version += 1
                    self._save_params_to_checkpoint_if_needed()
                    version = self._parameters.version
                    updated_version = True

            if updated_version:
                self._report_version_if_needed(version)
            res.version = version
            return res

    def wrap_optimizer(self):
        self._optimizer = OptimizerWrapper(
            self._optimizer,
            self._use_async,
            self._parameters.get_embedding_param,
            self._parameters.set_embedding_param,
        )

    def _report_version_if_needed(self, version):
        if self._eval_steps and version % self._eval_steps == 0:
            self._report_version(version)

    def _report_version(self, version):
        req = elasticdl_pb2.ReportVersionRequest()
        req.model_version = version
        self._master_stub.report_version(req)

    def wrap_optimizer_and_set_slot(self):
        if not self._use_wrap_opt:
            self.wrap_optimizer()
            self._parameters.create_slot_params(
                self._optimizer.allowed_slot_names,
                self._optimizer.slot_initial_value,
            )
            self._use_wrap_opt = True

    def _save_params_to_checkpoint_if_needed(self):
        """Save a checkpoint of parameters to a protobuf file"""
        if (
            self._checkpoint_saver
            and self._parameters.version % self._checkpoint_saver._steps == 0
        ):
            model_pb = self._parameters.to_model_pb()

            logger.info("Save checkpoint for version %s" % model_pb.version)
            self._checkpoint_saver.save(
                model_pb.version,
                model_pb,
                is_eval_checkpoint=False,
                shard_index=self._ps_id,
                shard_num=self._num_ps_pods,
            )
