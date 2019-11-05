import threading

from google.protobuf import empty_pb2

from elasticdl.proto import elasticdl_pb2, elasticdl_pb2_grpc
from elasticdl.python.common.tensor import (
    Tensor,
    emplace_tensor_pb_from_ndarray,
    serialize_tensor,
)


class PserverServicer(elasticdl_pb2_grpc.PserverServicer):
    """PS service implementation"""

    def __init__(
        self,
        parameters,
        grads_to_wait,
        optimizer,
        lr_staleness_modulation=False,
        use_async=False,
    ):
        self._parameters = parameters
        self._grads_to_wait = grads_to_wait
        self._optimizer = optimizer
        self._lr_staleness_modulation = lr_staleness_modulation
        self._use_async = use_async
        self._version_lock = threading.Lock()
        self._lock = threading.Lock()

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
            self._parameters.init_from_model_pb(request)
        return empty_pb2.Empty()

    def push_gradient(self, request, _):
        res = elasticdl_pb2.PushGradientResponse()
        if self._use_async:
            grad_vars = []
            for pb in request.gradients:
                grad = Tensor.from_tensor_pb(pb)
                self._parameters.check_grad(grad)
                var = self._parameters.get_non_embedding_param(grad.name)
                grad = grad.to_tf_tensor()
                grad_vars.append((grad, var))

            self._optimizer.apply_gradients(grad_vars)
            with self._version_lock:
                self._parameters.version += 1

            res.accepted = True
            res.model_version = self._parameters.version
            return res
        else:
            if request.model_version != self._parameters.version:
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

                if self._grads_n == self._grads_to_wait:
                    grad_vars = []
                    for name, grad in self._grads_buffer.items():
                        # Dense gradients are averaged,
                        # while sparse gradients are summed
                        if not grad.is_indexed_slices():
                            grad.values = grad.values / self._grads_to_wait
                        var = self._parameters.get_non_embedding_param(name)
                        grad_vars.append((grad.to_tf_tensor(), var))

                    self._optimizer.apply_gradients(grad_vars)
                    self._grads_n = 0
                    self._grads_buffer.clear()
                    self._parameters.version += 1

                res.model_version = self._parameters.version
                return res
