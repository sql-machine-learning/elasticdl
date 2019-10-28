import os
import unittest

import grpc
from google.protobuf import empty_pb2

from elasticdl.proto import elasticdl_pb2_grpc
from elasticdl.python.common.constants import GRPC
from elasticdl.python.ps.parameter_server import ParameterServer
from elasticdl.python.ps.parameters import Parameters

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
        self, parameters, grads_to_wait, lr_staleness_modulation, use_async
    ):
        args = PserverArgs(
            grads_to_wait=grads_to_wait,
            lr_staleness_modulation=lr_staleness_modulation,
            use_async=use_async,
            port=self._port,
        )
        pserver = ParameterServer(args)
        pserver.prepare()
        self._server = pserver.server
        self._stub = elasticdl_pb2_grpc.PserverStub(self._channel)

    def testServicer(self):
        parameters = Parameters()
        grads_to_wait = 8
        lr_staleness_modulation = False
        use_async = True

        self.create_server_and_stub(
            parameters, grads_to_wait, lr_staleness_modulation, use_async
        )

        # TODO: replace the section below with real RPC service tests
        # after service implementation
        try:
            req = empty_pb2.Empty()
            res = self._stub.pull_variable(req)
            self.assertEqual(res, req)
            res = self._stub.pull_embedding_vector(req)
            self.assertEqual(res, req)
            res = self._stub.push_model(req)
            self.assertEqual(res, req)
            res = self._stub.push_gradient(req)
            self.assertEqual(res, req)
        except Exception:
            self.assertTrue(False)
