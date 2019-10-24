import unittest

import numpy as np
import tensorflow as tf

from elasticdl.proto.elasticdl_pb2 import EmbeddingTableInfo
from elasticdl.python.ps.parameters import Parameters


class ParametersTest(unittest.TestCase):
    def setUp(self):
        self.params = Parameters()
        self.embedding_table_name = "embedding_1"
        self.embedding_dim = 10
        self.embedding_table_info_pb = EmbeddingTableInfo()
        self.embedding_table_info_pb.name = self.embedding_table_name
        self.embedding_table_info_pb.dim = self.embedding_dim
        self.embedding_table_info_pb.initializer = "uniform"

    def test_init(self):
        # TODO(qijun) waiting for Tensor/Model proto message definition
        pass

    def test_get_embedding_param(self):
        self.params.clear()

        self.params._init_embedding_param(self.embedding_table_info_pb)

        indices = [0, 3, 7]

        res = self.params.get_embedding_param(
            self.embedding_table_name, indices
        )
        self.assertTupleEqual(res.shape, (3, 10))

        res = self.params.get_embedding_param(self.embedding_table_name, [])
        self.assertIsNone(res)

        with self.assertRaises(ValueError):
            self.params.get_embedding_param("tom", indices)

    def test_set_embedding_param(self):
        self.params.clear()
        self.params._init_embedding_param(self.embedding_table_info_pb)
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
        row4 = self.params.get_embedding_param(self.embedding_table_name, [8])

        rows = [row0, row1, row4]
        rows = np.concatenate(rows)
        np.testing.assert_array_equal(rows, values)

        with self.assertRaises(ValueError):
            self.params.set_embedding_param("tom", [0, 1, 2], values)

    def test_non_embedding_params(self):
        self.params.clear()

        res = self.params.get_non_embedding_params()
        self.assertFalse(any(res))

        variables = {
            "x": tf.Variable(1, name="x"),
            "y": tf.Variable(2, name="y"),
        }

        self.params.set_non_embedding_params(variables)
        res = self.params.get_non_embedding_params()
        self.assertTrue("x" in res)
        self.assertTrue("y" in res)
