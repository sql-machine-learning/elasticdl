import threading

from google.protobuf import empty_pb2

from elasticdl.proto import elasticdl_pb2, elasticdl_pb2_grpc
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.common.tensor import (
    Tensor,
    emplace_tensor_pb_from_ndarray,
    serialize_tensor,
)
from elasticdl.python.master.optimizer_wrapper import OptimizerWrapper


class PserverServicer(elasticdl_pb2_grpc.PserverServicer):
    """PS service implementation"""

    def __init__(
        self,
        parameters,
        grads_to_wait,
        optimizer,
        lr_staleness_modulation=False,
        use_async=False,
        evaluation_steps=0,
        master_channel=None,
        checkpoint_service=None,
        ps_id=None,
        num_ps_pods=None,
        checkpoint_dir_for_init=None,
    ):
        if master_channel is None:
            self._master_stub = None
        else:
            self._master_stub = elasticdl_pb2_grpc.MasterStub(master_channel)

        self._parameters = parameters
        self._grads_to_wait = grads_to_wait
        self._optimizer = optimizer
        self._lr_staleness_modulation = lr_staleness_modulation
        self._use_async = use_async
        self._eval_steps = evaluation_steps
        self._checkpoint_service = checkpoint_service
        self._ps_id = ps_id
        self._num_ps_pods = num_ps_pods
        self._version_lock = threading.Lock()
        self._lock = threading.Lock()
        self._use_wrap_opt = False

        self._grads_n = 0
        self._grads_buffer = {}

    def pull_variable(self, request, _):
        """
        Response with all non-embedding parameters if initialized.
        """
        res = elasticdl_pb2.PullVariableResponse()
        if not self._parameters.init_status:
            res.model_init_status = False
            return res

        # Only sync-SGD needs lock
        # TODO: use a read-write lock to support multiple concurrent reads
        if not self._use_async:
            self._lock.acquire()
        res.model.version = self._parameters.version
        for name, var in self._parameters.non_embedding_params.items():
            emplace_tensor_pb_from_ndarray(
                res.model.param, var.numpy(), name=name
            )
        if not self._use_async:
            self._lock.release()
        res.model_init_status = True
        return res

    def pull_embedding_vector(self, request, _):
        ret = elasticdl_pb2.Tensor()
        if not request.ids:
            return ret
        embedding_vectors = self._parameters.get_embedding_param(
            request.name, request.ids
        )
        tensor = Tensor(values=embedding_vectors)
        serialize_tensor(tensor, ret)
        return ret

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

    def push_gradient(self, request, _):
        res = elasticdl_pb2.PushGradientResponse()
        if self._use_async:
            grad_vars = []
            for pb in request.gradients:
                grad = Tensor.from_tensor_pb(pb)
                self._parameters.check_grad(grad)
                name = grad.name
                var = self._parameters.get_non_embedding_param(name)
                grad = grad.to_tf_tensor()
                if var is None:
                    grad_vars.append((grad, name))
                else:
                    grad_vars.append((grad, var))

            self._optimizer.apply_gradients(grad_vars)
            with self._version_lock:
                self._parameters.version += 1
                self._save_params_to_checkpoint_if_needed()
                version = self._parameters.version
            self._report_version_if_needed(version)

            res.accepted = True
            res.model_version = self._parameters.version
            return res
        else:
            if request.model_version < self._parameters.version:
                res.accepted = False
                res.model_version = self._parameters.version
                return res

            with self._lock:
                for pb in request.gradients:
                    grad = Tensor.from_tensor_pb(pb)
                    self._parameters.check_grad(grad)
                    if grad.name in self._grads_buffer:
                        self._grads_buffer[grad.name] = (
                            self._grads_buffer[grad.name] + grad
                        )
                    else:
                        self._grads_buffer[grad.name] = grad

                self._grads_n += 1
                res.accepted = True

                updated_version = False
                version = self._parameters.version
                if self._grads_n == self._grads_to_wait:
                    grad_vars = []
                    for name, grad in self._grads_buffer.items():
                        # Dense gradients are averaged,
                        # while sparse gradients are summed
                        if not grad.is_indexed_slices():
                            grad.values = grad.values / self._grads_to_wait
                        var = self._parameters.get_non_embedding_param(name)
                        grad = grad.to_tf_tensor()
                        if var is None:
                            grad_vars.append((grad, name))
                        else:
                            grad_vars.append((grad, var))

                    self._optimizer.apply_gradients(grad_vars)
                    self._grads_n = 0
                    self._grads_buffer.clear()
                    self._parameters.version += 1
                    self._save_params_to_checkpoint_if_needed()
                    version = self._parameters.version
                    updated_version = True

            if updated_version:
                self._report_version_if_needed(version)
            res.model_version = version
            return res

    def wrap_optimizer(self):
        # TODO(yunjian.lmh): refine these arguments when we don't need
        # to support using Redis as distributed KV storage.
        embedding_dims = {}
        for table in self._parameters.embedding_params.values():
            embedding_dims[table.name] = table.dim
        embedding_service_endpoint = None

        def lookup_embedding_func(keys):
            embeddings = []
            for key in keys:
                arrs = key.split("-")
                layer_name = "-".join(arrs[:-1])
                id = int(arrs[-1])
                embedding = self._parameters.get_embedding_param(
                    layer_name, [id]
                )
                embeddings.append(embedding.flatten())
            return embeddings, []

        def update_embedding_func(keys, values):
            for key, value in zip(keys, values):
                arrs = key.split("-")
                layer_name = "-".join(arrs[:-1])
                id = int(arrs[-1])
                self._parameters.set_embedding_param(layer_name, [id], [value])

        self._optimizer = OptimizerWrapper(
            self._optimizer,
            embedding_service_endpoint,
            embedding_dims,
            self._use_async,
            lookup_embedding_func,
            update_embedding_func,
        )

    def _report_version_if_needed(self, version):
        if self._eval_steps and version % self._eval_steps == 0:
            self._report_version(version)

    def _report_version(self, version):
        req = elasticdl_pb2.ReportVersionRequest()
        req.model_version = version
        self._master_stub.ReportVersion(req)

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
            self._checkpoint_service
            and self._parameters.version % self._checkpoint_service._steps == 0
        ):
            model_pb = self._parameters.to_model_pb()

            logger.info("Save checkpoint for version %s" % model_pb.version)
            self._checkpoint_service.save(
                model_pb.version,
                model_pb,
                is_eval_checkpoint=False,
                shard_index=self._ps_id,
                shard_num=self._num_ps_pods,
            )
