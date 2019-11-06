import unittest

import numpy as np
import tensorflow as tf

from elasticdl.proto.elasticdl_pb2 import EmbeddingTableInfo, Model
from elasticdl.python.common.tensor import Tensor
from elasticdl.python.ps.parameters import Parameters


class ParametersTest(unittest.TestCase):
    def setUp(self):
        self.params = Parameters()

        self.model_pb = Model()
        self.tensors_pb = self.model_pb.param
        self.embeddings_pb = self.model_pb.embedding_table_info

        arr1 = np.random.uniform(size=(3, 4))
        tensor1_pb = Tensor(arr1, name="x").to_tensor_pb()
        arr2 = np.random.uniform(size=(4, 5))
        tensor2_pb = Tensor(arr2, name="y").to_tensor_pb()
        self.tensors_pb.extend([tensor1_pb, tensor2_pb])

        self.embedding_table_name = "embedding_1"
        self.embedding_dim = 10
        embedding_pb = EmbeddingTableInfo()
        embedding_pb.name = self.embedding_table_name
        embedding_pb.dim = self.embedding_dim
        embedding_pb.initializer = "uniform"

        self.embeddings_pb.append(embedding_pb)

    def _clear_params(self):
        self.params.non_embedding_params.clear()
        self.params.embedding_params.clear()

    def _test_get_embedding_param(self, slot_names=[], slot_init_value={}):
        indices = [0, 3, 7]

        res = self.params.get_embedding_param(
            self.embedding_table_name, indices
        )
        self.assertTupleEqual(res.shape, (3, 10))
        for slot in slot_names:
            res = self.params.get_embedding_param(
                self.embedding_table_name + "-" + slot, indices
            )
            self.assertTrue(((res - slot_init_value[slot]) < 0.0001).all())

        res = self.params.get_embedding_param(self.embedding_table_name, [])
        self.assertIsNone(res)

        with self.assertRaises(ValueError):
            self.params.get_embedding_param("tom", indices)

    def test_init_from_model_pb(self):
        self._clear_params()
        self.params.init_from_model_pb(self.model_pb)

        res = self.params.non_embedding_params
        self.assertTrue("x" in res)
        self.assertTrue("y" in res)
        self.assertTrue(res["x"].trainable)
        self.assertTupleEqual(tuple(res["y"].shape.as_list()), (4, 5))

        self._test_get_embedding_param()

    def test_non_embedding_params(self):
        self._clear_params()

        res = self.params.non_embedding_params
        self.assertFalse(any(res))

        variables = {
            "x": tf.Variable(1, name="x"),
            "y": tf.Variable(2, name="y"),
        }

        self.params.non_embedding_params = variables
        self.assertTrue("x" in self.params.non_embedding_params)
        self.assertTrue("y" in self.params.non_embedding_params)

    def test_get_embedding_param(self):
        self._clear_params()
        self.params._init_embedding_params(self.embeddings_pb)
        self._test_get_embedding_param()

    def test_set_embedding_param(self):
        self._clear_params()
        self.params._init_embedding_params(self.embeddings_pb)
        indices = [100, 34, 8]
        x = len(indices)
        values = np.random.uniform(size=x * self.embedding_dim).reshape(
            (x, self.embedding_dim)
        )

        self.params.set_embedding_param(
            self.embedding_table_name, indices, values
        )

        row0 = self.params.get_embedding_param(
            self.embedding_table_name, [100]
        )
        row1 = self.params.get_embedding_param(self.embedding_table_name, [34])
        row2 = self.params.get_embedding_param(self.embedding_table_name, [8])

        rows = [row0, row1, row2]
        rows = np.concatenate(rows)
        np.testing.assert_array_equal(rows, values)

        with self.assertRaises(ValueError):
            self.params.set_embedding_param("tom", [0, 1, 2], values)

    def test_check_grad(self):
        self._clear_params()
        self.params.init_from_model_pb(self.model_pb)

        grad0 = Tensor(name="z")
        with self.assertRaisesRegex(ValueError, "Name error"):
            self.params.check_grad(grad0)

        grad1 = Tensor(name="x", values=np.random.uniform(size=(3, 5)))
        with self.assertRaisesRegex(ValueError, "Non embedding param error"):
            self.params.check_grad(grad1)

        grad2 = Tensor(
            name="embedding_1",
            values=np.random.uniform(size=(3, 11)),
            indices=np.array([1, 2, 3]),
        )
        with self.assertRaisesRegex(
            ValueError, "ElasticDL embedding param error"
        ):
            self.params.check_grad(grad2)

        grad3 = Tensor(
            name="x",
            values=np.random.uniform(size=(4, 4)),
            indices=np.array([1, 2, 3, 4]),
        )
        with self.assertRaisesRegex(ValueError, "Keras embedding param error"):
            self.params.check_grad(grad3)

    def test_create_slot_params(self):
        # At first, no embedding table are in the parameters
        self.assertFalse(self.params.has_embedding_params())

        # create embedding tables in the parameters
        self.params._init_embedding_params(self.embeddings_pb)
        self.assertTrue(self.params.has_embedding_params())

        slot_names = ["accumulator", "linear"]
        slot_init_value = {
            slot_names[0]: 3.5,
            slot_names[1]: 0.0,
        }
        self.params.create_slot_params(slot_names, slot_init_value)
        self._test_get_embedding_param(slot_names, slot_init_value)
