import threading

from google.protobuf import empty_pb2

from elasticdl.proto import elasticdl_pb2, elasticdl_pb2_grpc


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
        self._version = 0
        self._lock = threading.Lock()

    def pull_variable(self, request, _):
        # TODO: implement this RPC service
        return elasticdl_pb2.PullVariableResponse()

    def pull_embedding_vector(self, request, _):
        # TODO: implement this RPC service
        return elasticdl_pb2.Tensor()

    def push_model(self, request, _):
        with self._lock:
            self._parameters.init_from_model_pb(request)
        return empty_pb2.Empty()

    def push_gradient(self, request, _):
        # TODO: implement this RPC service
        return elasticdl_pb2.PushGradientResponse()
