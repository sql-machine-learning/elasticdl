import os
import unittest

import grpc
import numpy as np
from google.protobuf import empty_pb2

from elasticdl.proto import elasticdl_pb2, elasticdl_pb2_grpc
from elasticdl.python.common.constants import GRPC
from elasticdl.python.common.tensor import emplace_tensor_pb_from_ndarray
from elasticdl.python.ps.parameter_server import ParameterServer

_test_model_zoo_path = os.path.dirname(os.path.realpath(__file__))


class PserverArgs(object):
    def __init__(
        self,
        grads_to_wait=8,
        lr_staleness_modulation=0,
        use_async=False,
        model_zoo=_test_model_zoo_path,
        model_def="test_module.custom_model",
        optimizer="optimizer",
        port=9999,
        log_level="INFO",
    ):
        self.grads_to_wait = grads_to_wait
        self.lr_staleness_modulation = lr_staleness_modulation
        self.use_async = use_async
        self.model_zoo = model_zoo
        self.model_def = model_def
        self.optimizer = optimizer
        self.port = port
        self.log_level = log_level


class PserverServicerTest(unittest.TestCase):
    def setUp(self):
        self._port = 9999
        addr = "localhost:%d" % self._port
        self._channel = grpc.insecure_channel(
            addr,
            options=[
                ("grpc.max_send_message_length", GRPC.MAX_SEND_MESSAGE_LENGTH),
                (
                    "grpc.max_receive_message_length",
                    GRPC.MAX_RECEIVE_MESSAGE_LENGTH,
                ),
            ],
        )
        self._server = None

    def tearDown(self):
        if self._server:
            self._server.stop(0)

    def create_server_and_stub(
        self, grads_to_wait, lr_staleness_modulation, use_async
    ):
        args = PserverArgs(
            grads_to_wait=grads_to_wait,
            lr_staleness_modulation=lr_staleness_modulation,
            use_async=use_async,
            port=self._port,
        )
        pserver = ParameterServer(args)
        pserver.prepare()
        self._parameters = pserver.parameters
        self._server = pserver.server
        self._stub = elasticdl_pb2_grpc.PserverStub(self._channel)

    def create_default_server_and_stub(self):
        grads_to_wait = 8
        lr_staleness_modulation = False
        use_async = True

        self.create_server_and_stub(
            grads_to_wait, lr_staleness_modulation, use_async
        )

    def testServicer(self):
        self.create_default_server_and_stub()

        # TODO: replace the section below with real RPC service tests
        # after service implementation
        req = elasticdl_pb2.PullVariableRequest()
        res = self._stub.pull_variable(req)
        self.assertEqual(res, elasticdl_pb2.PullVariableResponse())

        req = elasticdl_pb2.PullEmbeddingVectorRequest()
        res = self._stub.pull_embedding_vector(req)
        self.assertEqual(res, elasticdl_pb2.Tensor())

        req = elasticdl_pb2.PushGradientRequest()
        res = self._stub.push_gradient(req)
        self.assertEqual(res, elasticdl_pb2.PushGradientResponse())

    def testPushModel(self):
        self.create_default_server_and_stub()
        param0 = {
            "v0": np.random.rand(3, 2).astype(np.float32),
            "v1": np.random.rand(10, 32).astype(np.float32),
        }
        param1 = {
            "v0": np.ones([3, 2], dtype=np.float32),
            "v1": np.ones([10, 32], dtype=np.float32),
        }
        embedding_info = elasticdl_pb2.EmbeddingTableInfo()
        embedding_info.name = "layer0"
        embedding_info.dim = 32
        embedding_info.initializer = "normal"

        models = [param0, param1]

        for idx, model in enumerate(models):
            req = elasticdl_pb2.Model()
            req.version = idx + 1
            for name in model:
                emplace_tensor_pb_from_ndarray(
                    req.param, model[name], name=name
                )
            req.embedding_table_info.append(embedding_info)
            res = self._stub.push_model(req)
            self.assertEqual(res, empty_pb2.Empty())
            # self._parameters is initialized with the first push_model call
            # and the second push_model has no effect
            self.assertEqual(self._parameters.version, 1)
            for name in param0:
                self.assertTrue(
                    np.allclose(
                        param0[name],
                        self._parameters.non_embedding_params[name].numpy(),
                    )
                )
            self.assertEqual(
                embedding_info.name,
                self._parameters.embedding_params[embedding_info.name].name,
            )
            self.assertEqual(
                embedding_info.dim,
                self._parameters.embedding_params[embedding_info.name].dim,
            )
            self.assertEqual(
                embedding_info.initializer,
                self._parameters.embedding_params[
                    embedding_info.name
                ].initializer,
            )
