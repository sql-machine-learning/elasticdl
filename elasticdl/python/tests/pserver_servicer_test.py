import os
import unittest

import grpc
import numpy as np
import tensorflow as tf
from google.protobuf import empty_pb2

from elasticdl.proto import elasticdl_pb2, elasticdl_pb2_grpc
from elasticdl.python.common.constants import GRPC
from elasticdl.python.common.tensor import (
    emplace_tensor_pb_from_ndarray,
    tensor_pb_to_ndarray,
)
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
        embedding_info = elasticdl_pb2.EmbeddingTableInfo()
        embedding_info.name = "layer_a"
        embedding_info.dim = 32
        embedding_info.initializer = "normal"
        self._embedding_info = embedding_info
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

        self._lr = 0.1

    def create_default_server_and_stub(self):
        grads_to_wait = 8
        lr_staleness_modulation = False
        use_async = True

        self.create_server_and_stub(
            grads_to_wait, lr_staleness_modulation, use_async
        )

    def get_embedding_vectors(self, name, ids):
        pull_req = elasticdl_pb2.PullEmbeddingVectorRequest()
        pull_req.name = name
        pull_req.ids.extend(ids)
        res = self._stub.pull_embedding_vector(pull_req)
        if res.content:
            return tensor_pb_to_ndarray(res)
        else:
            return None

    def test_push_model(self):
        self.create_default_server_and_stub()
        param0 = {
            "v0": np.random.rand(3, 2).astype(np.float32),
            "v1": np.random.rand(10, 32).astype(np.float32),
        }
        param1 = {
            "v0": np.ones([3, 2], dtype=np.float32),
            "v1": np.ones([10, 32], dtype=np.float32),
        }

        models = [param0, param1]

        for idx, model in enumerate(models):
            req = elasticdl_pb2.Model()
            req.version = idx + 1
            for name in model:
                emplace_tensor_pb_from_ndarray(
                    req.param, model[name], name=name
                )
            req.embedding_table_info.append(self._embedding_info)
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
                self._embedding_info.name,
                self._parameters.embedding_params[
                    self._embedding_info.name
                ].name,
            )
            self.assertEqual(
                self._embedding_info.dim,
                self._parameters.embedding_params[
                    self._embedding_info.name
                ].dim,
            )
            self.assertEqual(
                self._embedding_info.initializer,
                self._parameters.embedding_params[
                    self._embedding_info.name
                ].initializer,
            )

    def test_pull_variable(self):
        self.create_default_server_and_stub()
        param0 = {
            "v0": np.random.rand(3, 2).astype(np.float32),
            "v1": np.random.rand(10, 32).astype(np.float32),
        }
        pull_req = empty_pb2.Empty()
        # try to pull variable
        res = self._stub.pull_variable(pull_req)
        # not initialized
        self.assertFalse(res.model_init_status)

        # init variable
        req = elasticdl_pb2.Model()
        req.version = 1
        for name, var in param0.items():
            emplace_tensor_pb_from_ndarray(req.param, var, name=name)
        res = self._stub.push_model(req)
        self.assertEqual(res, empty_pb2.Empty())

        # pull variable back
        res = self._stub.pull_variable(pull_req)
        self.assertTrue(res.model_init_status)
        self.assertEqual(res.model.version, req.version)
        for param in res.model.param:
            name = param.name
            tensor = tensor_pb_to_ndarray(param)
            self.assertTrue(np.allclose(param0[name], tensor))

    def test_pull_embedding_vector(self):
        self.create_default_server_and_stub()

        id_list_0 = [1, 3, 9, 6]
        id_list_1 = [8, 9, 1, 0, 6]

        req = elasticdl_pb2.Model()
        req.version = 1
        req.embedding_table_info.append(self._embedding_info)
        another_embedding_info = elasticdl_pb2.EmbeddingTableInfo()
        another_embedding_info.name = "layer_b"
        another_embedding_info.dim = 16
        another_embedding_info.initializer = "normal"
        req.embedding_table_info.append(another_embedding_info)
        res = self._stub.push_model(req)
        self.assertEqual(res, empty_pb2.Empty())

        vectors_a_0 = self.get_embedding_vectors("layer_a", id_list_0)
        self.assertEqual(vectors_a_0.shape[0], len(id_list_0))
        self.assertEqual(vectors_a_0.shape[1], 32)

        vectors_a_1 = self.get_embedding_vectors("layer_a", id_list_1)
        self.assertEqual(vectors_a_1.shape[0], len(id_list_1))
        self.assertEqual(vectors_a_1.shape[1], 32)

        vectors_b_1 = self.get_embedding_vectors("layer_b", id_list_1)
        self.assertEqual(vectors_b_1.shape[0], len(id_list_1))
        self.assertEqual(vectors_b_1.shape[1], 16)

        vectors_b_0 = self.get_embedding_vectors("layer_b", id_list_0)
        self.assertEqual(vectors_b_0.shape[0], len(id_list_0))
        self.assertEqual(vectors_b_0.shape[1], 16)

        for idx0, id0 in enumerate(id_list_0):
            for idx1, id1 in enumerate(id_list_1):
                if id0 == id1:
                    self.assertTrue(
                        np.array_equal(vectors_a_0[idx0], vectors_a_1[idx1])
                    )
                    self.assertTrue(
                        np.array_equal(vectors_b_0[idx0], vectors_b_1[idx1])
                    )

        vectors = self.get_embedding_vectors("layer_a", [])
        self.assertEqual(vectors, None)

    def test_push_gradient_async_update(self):
        self.create_default_server_and_stub()

        var_name = ["test_1", "test_2"]
        var_value = [
            np.array([10.0, 20.0, 30.0], np.float32),
            np.array([20.0, 40.0, 60.0], np.float32),
        ]
        grad_value = [
            np.array([1.0, 2.0, 3.0], np.float32),
            np.array([2.0, 4.0, 6.0], np.float32),
        ]
        non_embedding_params = self._parameters.non_embedding_params
        for i in range(2):
            non_embedding_params[var_name[i]] = tf.Variable(var_value[i])

        req = elasticdl_pb2.PushGradientRequest()
        for g, name in zip(grad_value, var_name):
            emplace_tensor_pb_from_ndarray(req.gradients, g, name=name)
        res = self._stub.push_gradient(req)
        self.assertEqual(res.accepted, True)
        self.assertEqual(res.model_version, 1)
        for idx, name in enumerate(var_name):
            expected_value = var_value[idx] - self._lr * grad_value[idx]
            self.assertTrue(
                np.allclose(expected_value, non_embedding_params[name].numpy())
            )

        # Test applying gradients with same name
        req = elasticdl_pb2.PushGradientRequest()
        for g in grad_value:
            emplace_tensor_pb_from_ndarray(req.gradients, g, name=var_name[0])
        res = self._stub.push_gradient(req)
        self.assertEqual(res.accepted, True)
        self.assertEqual(res.model_version, 2)
        expected_values = [
            var_value[0]
            - self._lr * grad_value[0] * 2
            - self._lr * grad_value[1],
            var_value[1] - self._lr * grad_value[1],
        ]
        for expected_value, name in zip(expected_values, var_name):
            self.assertTrue(
                np.allclose(expected_value, non_embedding_params[name].numpy())
            )

        # TODO: test applying gradients with embedding params

    def test_push_gradient_sync_update(self):
        self.create_server_and_stub(
            grads_to_wait=2, lr_staleness_modulation=False, use_async=False
        )
        var_name = ["test_1", "test_2"]
        var_value = [
            np.array([10.0, 20.0, 30.0], np.float32),
            np.array([20.0, 40.0, 60.0], np.float32),
        ]

        non_embedding_params = self._parameters.non_embedding_params
        for i in range(2):
            non_embedding_params[var_name[i]] = tf.Variable(var_value[i])

        grad_value0 = [
            np.array([1.0, 2.0, 3.0], np.float32),
            np.array([2.0, 4.0, 6.0], np.float32),
        ]
        grad_value1 = [
            np.array([0.0, 0.0, 7.0], np.float32),
            np.array([9.0, 9.0, 6.0], np.float32),
        ]

        req = elasticdl_pb2.PushGradientRequest()
        req.model_version = 0
        for g, name in zip(grad_value0, var_name):
            emplace_tensor_pb_from_ndarray(req.gradients, g, name=name)
        res = self._stub.push_gradient(req)
        self.assertEqual(res.accepted, True)
        self.assertEqual(res.model_version, 0)

        req = elasticdl_pb2.PushGradientRequest()
        req.model_version = 0
        for g, name in zip(grad_value1, var_name):
            emplace_tensor_pb_from_ndarray(req.gradients, g, name=name)
        res = self._stub.push_gradient(req)
        self.assertEqual(res.accepted, True)
        self.assertEqual(res.model_version, 1)

        req = elasticdl_pb2.PushGradientRequest()
        req.model_version = 0
        for g, name in zip(grad_value1, var_name):
            emplace_tensor_pb_from_ndarray(req.gradients, g, name=name)
        res = self._stub.push_gradient(req)
        self.assertEqual(res.accepted, False)
        self.assertEqual(res.model_version, 1)

        expected_values = [
            var_value[0] - self._lr * (grad_value0[0] + grad_value1[0]) / 2,
            var_value[1] - self._lr * (grad_value0[1] + grad_value1[1]) / 2,
        ]
        for expected_value, name in zip(expected_values, var_name):
            self.assertTrue(
                np.allclose(expected_value, non_embedding_params[name].numpy())
            )

        # TODO: test applying gradients with embedding params
