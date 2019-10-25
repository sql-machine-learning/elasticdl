import unittest
from concurrent import futures

import grpc
import tensorflow as tf
from google.protobuf import empty_pb2

from elasticdl.proto import elasticdl_pb2, elasticdl_pb2_grpc
from elasticdl.python.common.constants import GRPC
from elasticdl.python.ps.parameters import Parameters
from elasticdl.python.ps.servicer import PserverServicer


class PserverServicerTest(unittest.TestCase):
    @staticmethod
    def create_server(
        parameters,
        grads_to_wait,
        optimizer,
        lr_staleness_modulation,
        use_async,
        port,
    ):
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=64),
            options=[
                ("grpc.max_send_message_length", GRPC.MAX_SEND_MESSAGE_LENGTH),
                (
                    "grpc.max_receive_message_length",
                    GRPC.MAX_RECEIVE_MESSAGE_LENGTH,
                ),
            ],
        )

        pserver_servicer = PserverServicer(
            parameters,
            grads_to_wait,
            optimizer,
            lr_staleness_modulation=lr_staleness_modulation,
            use_async=use_async,
        )

        elasticdl_pb2_grpc.add_PserverServicer_to_server(
            pserver_servicer, server
        )
        server.add_insecure_port("[::]:{}".format(port))
        return server

    @staticmethod
    def create_stub(port):
        addr = "localhost:%d" % port
        channel = grpc.insecure_channel(
            addr,
            options=[
                ("grpc.max_send_message_length", GRPC.MAX_SEND_MESSAGE_LENGTH),
                (
                    "grpc.max_receive_message_length",
                    GRPC.MAX_RECEIVE_MESSAGE_LENGTH,
                ),
            ],
        )
        stub = elasticdl_pb2_grpc.PserverStub(channel)
        return stub

    def testServicer(self):
        parameters = Parameters()
        grads_to_wait = 8
        optimizer = tf.keras.optimizers.SGD()
        lr_staleness_modulation = False
        use_async = True
        port = 9999

        server = self.create_server(
            parameters,
            grads_to_wait,
            optimizer,
            lr_staleness_modulation,
            use_async,
            port,
        )
        server.start()

        stub = self.create_stub(port)

        # TODO: replace the section below with real RPC service tests
        # after service implementation
        try:
            req = empty_pb2.Empty()
            res = stub.pull_variable(req)
            self.assertEqual(res, req)
            res = stub.pull_embedding_vector(req)
            self.assertEqual(res, req)
            res = stub.push_model(req)
            self.assertEqual(res, req)
            res = stub.push_gradient(req)
            self.assertEqual(res, req)
        except Exception:
            self.assertTrue(False)

        server.stop(0)
