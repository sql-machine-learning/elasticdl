import os
import tempfile
import unittest

import numpy as np
import tensorflow as tf
from google.protobuf import empty_pb2

from elasticdl.proto import elasticdl_pb2, elasticdl_pb2_grpc
from elasticdl.python.common.grpc_utils import build_channel
from elasticdl.python.common.model_utils import (
    get_module_file_path,
    load_module,
)
from elasticdl.python.common.save_utils import CheckpointSaver
from elasticdl.python.common.tensor_utils import (
    pb_to_ndarray,
    serialize_indexed_slices,
    serialize_ndarray,
)
from elasticdl.python.ps.embedding_table import (
    EmbeddingTable,
    get_slot_table_name,
)
from elasticdl.python.ps.parameter_server import ParameterServer
from elasticdl.python.ps.parameters import Parameters
from elasticdl.python.ps.servicer import PserverServicer
from elasticdl.python.tests.test_utils import PserverArgs

_test_model_zoo_path = os.path.dirname(os.path.realpath(__file__))
_module_file = get_module_file_path(
    _test_model_zoo_path, "test_module.custom_model"
)


class PserverServicerTest(unittest.TestCase):
    def setUp(self):
        self._port = 9999
        addr = "localhost:%d" % self._port
        self._channel = build_channel(addr)
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
        self, grads_to_wait, lr_staleness_modulation, use_async, **kwargs
    ):
        args = PserverArgs(
            grads_to_wait=grads_to_wait,
            lr_staleness_modulation=lr_staleness_modulation,
            use_async=use_async,
            port=self._port,
            model_zoo=_test_model_zoo_path,
            model_def="test_module.custom_model",
            **kwargs
        )
        pserver = ParameterServer(args)
        pserver.prepare()
        self._parameters = pserver.parameters
        self._server = pserver.server
        self._stub = elasticdl_pb2_grpc.PserverStub(self._channel)

        self._lr = 0.1

    def create_default_server_and_stub(self, **kwargs):
        grads_to_wait = 8
        lr_staleness_modulation = False
        use_async = True

        self.create_server_and_stub(
            grads_to_wait, lr_staleness_modulation, use_async, **kwargs
        )

    def get_embedding_vectors(self, name, ids):
        pull_req = elasticdl_pb2.PullEmbeddingVectorRequest()
        pull_req.name = name
        pull_req.ids.extend(ids)
        res = self._stub.pull_embedding_vectors(pull_req)
        if res.tensor_content:
            return pb_to_ndarray(res)
        else:
            return None

    def test_push_model(self):
        opt_func_name = "ftrl_optimizer"
        opt = load_module(_module_file).__dict__[opt_func_name]()
        opt_config = opt.get_config()
        slot_names = ["accumulator", "linear"]
        slot_init_value = {
            "accumulator": opt_config["initial_accumulator_value"],
            "linear": 0.0,
        }

        self.create_default_server_and_stub(optimizer=opt_func_name)
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
                serialize_ndarray(model[name], req.dense_parameters[name])
            req.embedding_table_infos.append(self._embedding_info)
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
                tf.keras.initializers.get(
                    self._embedding_info.initializer
                ).__class__,
                self._parameters.embedding_params[
                    self._embedding_info.name
                ].initializer.__class__,
            )

            for slot_name in slot_names:
                name = get_slot_table_name(
                    self._embedding_info.name, slot_name
                )
                table = self._parameters.embedding_params[name]
                self.assertTrue(name, table.name)
                self.assertTrue(self._embedding_info.dim, table.dim)
                embedding = table.get([2])
                self.assertTrue(
                    (embedding - slot_init_value[slot_name] < 0.0001).all()
                )

    def test_pull_dense_parameters(self):
        self.create_default_server_and_stub()
        param0 = {
            "v0": np.random.rand(3, 2).astype(np.float32),
            "v1": np.random.rand(10, 32).astype(np.float32),
        }
        pull_req = elasticdl_pb2.PullDenseParametersRequest()
        pull_req.version = -1
        # try to pull variable
        res = self._stub.pull_dense_parameters(pull_req)
        # not initialized
        self.assertFalse(res.initialized)

        # init variable
        req = elasticdl_pb2.Model()
        req.version = 1
        for name, var in param0.items():
            serialize_ndarray(var, req.dense_parameters[name])
        res = self._stub.push_model(req)
        self.assertEqual(res, empty_pb2.Empty())

        # pull variable back
        res = self._stub.pull_dense_parameters(pull_req)
        self.assertTrue(res.initialized)
        self.assertEqual(res.version, req.version)
        for name, pb in res.dense_parameters.items():
            tensor = pb_to_ndarray(pb)
            self.assertTrue(np.allclose(param0[name], tensor))

        # pull variable again, no param as no updated version
        pull_req.version = res.version
        res = self._stub.pull_dense_parameters(pull_req)
        self.assertTrue(res.initialized)
        self.assertEqual(res.version, pull_req.version)
        self.assertTrue(not res.dense_parameters)

    def test_pull_embedding_vectors(self):
        self.create_default_server_and_stub()

        id_list_0 = [1, 3, 9, 6]
        id_list_1 = [8, 9, 1, 0, 6]

        req = elasticdl_pb2.Model()
        req.version = 1
        req.embedding_table_infos.append(self._embedding_info)
        another_embedding_info = elasticdl_pb2.EmbeddingTableInfo()
        another_embedding_info.name = "layer_b"
        another_embedding_info.dim = 16
        another_embedding_info.initializer = "normal"
        req.embedding_table_infos.append(another_embedding_info)
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

    def push_gradient_test_setup(self):
        self.var_names = ["test_1", "test_2"]
        self.var_values = [
            np.array([10.0, 20.0, 30.0], np.float32),
            np.array([20.0, 40.0, 60.0], np.float32),
        ]
        self.grad_values0 = [
            np.array([1.0, 2.0, 3.0], np.float32),
            np.array([2.0, 4.0, 6.0], np.float32),
        ]
        self.grad_values1 = [
            np.array([0.0, 0.0, 7.0], np.float32),
            np.array([9.0, 9.0, 6.0], np.float32),
        ]

        dim = self._embedding_info.dim
        self.embedding_table = (
            np.random.rand(4 * dim).reshape((4, dim)).astype(np.float32)
        )
        self.embedding_grads0 = tf.IndexedSlices(
            values=np.random.rand(3 * dim)
            .reshape((3, dim))
            .astype(np.float32),
            indices=(3, 1, 3),
        )
        self.embedding_grads1 = tf.IndexedSlices(
            values=np.random.rand(3 * dim)
            .reshape((3, dim))
            .astype(np.float32),
            indices=(2, 2, 3),
        )
        push_model_req = elasticdl_pb2.Model()
        push_model_req.version = self._parameters.version
        for name, value in zip(self.var_names, self.var_values):
            serialize_ndarray(value, push_model_req.dense_parameters[name])
        push_model_req.embedding_table_infos.append(self._embedding_info)
        self._stub.push_model(push_model_req)

        for name, var in zip(self.var_names, self.var_values):
            self._parameters.non_embedding_params[name] = tf.Variable(var)

        self._parameters.embedding_params[self._embedding_info.name].set(
            range(len(self.embedding_table)), self.embedding_table
        )

    def test_push_gradient_async_update(self):
        self.create_default_server_and_stub()
        self.push_gradient_test_setup()

        # Test applying gradients to embedding and non-embedding parameters
        req = elasticdl_pb2.PushGradientsRequest()
        for g, name in zip(self.grad_values0, self.var_names):
            serialize_ndarray(g, req.gradients.dense_parameters[name])
        serialize_indexed_slices(
            self.embedding_grads0,
            req.gradients.embedding_tables[self._embedding_info.name],
        )
        res = self._stub.push_gradients(req)
        self.assertEqual(res.accepted, True)
        self.assertEqual(res.version, 1)
        expected_values = [
            v - self._lr * g
            for v, g in zip(self.var_values, self.grad_values0)
        ]
        for name, expected_value in zip(self.var_names, expected_values):
            self.assertTrue(
                np.allclose(
                    expected_value,
                    self._parameters.non_embedding_params[name].numpy(),
                )
            )

        expected_embed_table = np.copy(self.embedding_table)
        for gv, gi in zip(
            self.embedding_grads0.values, self.embedding_grads0.indices
        ):
            expected_embed_table[gi] -= self._lr * gv

        actual_embed_table = self._parameters.get_embedding_param(
            self._embedding_info.name, range(len(expected_embed_table))
        )
        self.assertTrue(np.allclose(expected_embed_table, actual_embed_table))

        # Test applying gradients with same name
        for name, var in zip(self.var_names, self.var_values):
            self._parameters.non_embedding_params[name] = tf.Variable(var)
        req = elasticdl_pb2.PushGradientsRequest()
        serialize_ndarray(
            self.grad_values1[1],
            req.gradients.dense_parameters[self.var_names[0]],
        )
        res = self._stub.push_gradients(req)
        self.assertEqual(res.accepted, True)
        self.assertEqual(res.version, 2)
        expected_values = [
            self.var_values[0] - self._lr * self.grad_values1[1],
            self.var_values[1],
        ]
        for expected_value, name in zip(expected_values, self.var_names):
            self.assertTrue(
                np.allclose(
                    expected_value,
                    self._parameters.non_embedding_params[name].numpy(),
                )
            )

    def test_push_gradient_sync_update(self):
        self.create_server_and_stub(
            grads_to_wait=2, lr_staleness_modulation=False, use_async=False
        )
        self.push_gradient_test_setup()

        req = elasticdl_pb2.PushGradientsRequest()
        req.gradients.version = 0
        for g, name in zip(self.grad_values0, self.var_names):
            serialize_ndarray(g, req.gradients.dense_parameters[name])
        serialize_indexed_slices(
            self.embedding_grads0,
            req.gradients.embedding_tables[self._embedding_info.name],
        )

        res = self._stub.push_gradients(req)
        self.assertEqual(res.accepted, True)
        self.assertEqual(res.version, 0)

        req = elasticdl_pb2.PushGradientsRequest()
        req.gradients.version = 0
        for g, name in zip(self.grad_values1, self.var_names):
            serialize_ndarray(g, req.gradients.dense_parameters[name])
        serialize_indexed_slices(
            self.embedding_grads1,
            req.gradients.embedding_tables[self._embedding_info.name],
        )
        res = self._stub.push_gradients(req)
        self.assertEqual(res.accepted, True)
        self.assertEqual(res.version, 1)

        req = elasticdl_pb2.PushGradientsRequest()
        req.gradients.version = 0
        for g, name in zip(self.grad_values1, self.var_names):
            serialize_ndarray(g, req.gradients.dense_parameters[name])
        res = self._stub.push_gradients(req)
        self.assertEqual(res.accepted, False)
        self.assertEqual(res.version, 1)

        expected_values = [
            self.var_values[0]
            - self._lr * (self.grad_values0[0] + self.grad_values1[0]) / 2,
            self.var_values[1]
            - self._lr * (self.grad_values0[1] + self.grad_values1[1]) / 2,
        ]
        for expected_value, name in zip(expected_values, self.var_names):
            self.assertTrue(
                np.allclose(
                    expected_value,
                    self._parameters.non_embedding_params[name].numpy(),
                )
            )

        expected_embed_table = np.copy(self.embedding_table)
        for gv, gi in zip(
            self.embedding_grads0.values, self.embedding_grads0.indices
        ):
            expected_embed_table[gi] -= self._lr * gv
        for gv, gi in zip(
            self.embedding_grads1.values, self.embedding_grads1.indices
        ):
            expected_embed_table[gi] -= self._lr * gv

        actual_embed_table = self._parameters.get_embedding_param(
            self._embedding_info.name, range(len(expected_embed_table))
        )
        self.assertTrue(np.allclose(expected_embed_table, actual_embed_table))

    def test_save_parameters_to_checkpoint_file(self):
        with tempfile.TemporaryDirectory() as tempdir:
            checkpoint_saver = CheckpointSaver(
                checkpoint_dir=os.path.join(tempdir, "ckpt/"),
                checkpoint_steps=5,
                keep_checkpoint_max=3,
                include_evaluation=False,
            )
            pserver_servicer = PserverServicer(
                parameters=Parameters(),
                grads_to_wait=0,
                optimizer="optimizer",
                checkpoint_saver=checkpoint_saver,
                ps_id=0,
                num_ps_pods=1,
            )
            model_params = {
                "v0": tf.Variable([[1, 1, 1], [1, 1, 1]]),
                "v1": tf.Variable([[2, 2, 2], [2, 2, 2]]),
            }

            server_params = pserver_servicer._parameters
            for var_name, var_value in model_params.items():
                server_params.non_embedding_params[var_name] = var_value

            embedding_table = EmbeddingTable(
                name="embedding_0", dim=3, initializer="random_uniform"
            )
            server_params.embedding_params["embedding_0"] = embedding_table
            server_params.set_embedding_param(
                name="embedding_0",
                indices=np.array([0, 1]),
                values=np.array([[1, 1, 1], [2, 2, 2]]),
            )

            for i in range(100):
                pserver_servicer._parameters.version += 1
                pserver_servicer._save_params_to_checkpoint_if_needed()

            self.assertEqual(len(os.listdir(checkpoint_saver._directory)), 3)
            self.assertEqual(
                sorted(os.listdir(checkpoint_saver._directory)),
                ["version-100", "version-90", "version-95"],
            )
            self.assertEqual(
                os.listdir(checkpoint_saver._directory + "/version-100"),
                ["variables-0-of-1.ckpt"],
            )

    def test_restore_parameters_from_checkpoint(self):
        checkpoint_dir = "elasticdl/python/tests/testdata/ps_ckpt"
        checkpoint_saver = CheckpointSaver(checkpoint_dir, 0, 0, False)
        params = Parameters()
        table = EmbeddingTable("embedding", 2, "random_uniform")
        table.set([0, 1, 2, 3], np.ones((4, 2), dtype=np.float32))
        params.embedding_params["embedding"] = table
        params.non_embedding_params["dense/kernel:0"] = tf.Variable(
            [[1.0], [1.0]]
        )
        params.non_embedding_params["dense/bias:0"] = tf.Variable([1.0])
        params.version = 100
        model_pb = params.to_model_pb()
        checkpoint_saver.save(100, model_pb, False)

        checkpoint_dir_for_init = checkpoint_dir + "/version-100"
        args = PserverArgs(
            ps_id=0,
            num_ps_pods=2,
            model_zoo=_test_model_zoo_path,
            model_def="test_module.custom_model",
            checkpoint_dir_for_init=checkpoint_dir_for_init,
        )
        pserver_0 = ParameterServer(args)

        embedding_table = pserver_0.parameters.embedding_params["embedding"]
        self.assertEqual(
            list(embedding_table.embedding_vectors.keys()), [0, 2]
        )
        self.assertEqual(
            list(pserver_0.parameters.non_embedding_params.keys()),
            ["dense/kernel:0"],
        )
        self.assertTrue(
            np.array_equal(
                pserver_0.parameters.non_embedding_params[
                    "dense/kernel:0"
                ].numpy(),
                np.array([[1], [1]], dtype=int),
            )
        )
        self.assertEqual(pserver_0.parameters.version, 100)

        args = PserverArgs(
            ps_id=1,
            num_ps_pods=2,
            model_zoo=_test_model_zoo_path,
            model_def="test_module.custom_model",
            checkpoint_dir_for_init=checkpoint_dir_for_init,
        )
        pserver_1 = ParameterServer(args)

        embedding_table = pserver_1.parameters.embedding_params["embedding"]
        self.assertEqual(
            list(embedding_table.embedding_vectors.keys()), [1, 3]
        )
        self.assertEqual(
            list(pserver_1.parameters.non_embedding_params.keys()),
            ["dense/bias:0"],
        )
        self.assertTrue(
            np.array_equal(
                pserver_1.parameters.non_embedding_params[
                    "dense/bias:0"
                ].numpy(),
                np.array([1], dtype=int),
            )
        )
        self.assertEqual(pserver_1.parameters.version, 100)


if __name__ == "__main__":
    unittest.main()
