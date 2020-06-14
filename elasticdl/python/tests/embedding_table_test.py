# Copyright 2020 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import tensorflow as tf

from elasticdl.proto.elasticdl_pb2 import EmbeddingTableInfo
from elasticdl.python.ps.embedding_table import (
    EmbeddingTable,
    create_embedding_table,
    get_slot_table_name,
)


class EmbeddingTableTest(unittest.TestCase):
    def setUp(self):
        self.name = "embedding_1"
        self.dim = 10
        self.initializer = "uniform"
        self.table = EmbeddingTable(self.name, self.dim, self.initializer)

    def test_embedding_table_init(self):
        self.assertIsNotNone(self.table)
        self.assertEqual(self.table.name, self.name)
        self.assertEqual(self.table.dim, self.dim)
        self.assertEqual(
            tf.keras.initializers.get(self.initializer).__class__,
            self.table.initializer.__class__,
        )

    def test_embedding_table_get(self):
        self.table.clear()
        indices = [0, 3, 7]
        res = self.table.get(indices)
        self.assertTupleEqual(res.shape, (3, 10))

        res = self.table.get([])
        self.assertIsNone(res)

        self.table.get([0, 3, 8])
        self.assertEqual(len(self.table.embedding_vectors), 4)

    def test_embedding_table_set(self):
        self.table.clear()
        indices = [0, 1, 4]
        x = len(indices)
        values = np.random.uniform(size=x * self.dim).reshape((x, self.dim))
        self.table.set(indices, values)

        row0 = self.table.get([0])
        row1 = self.table.get([1])
        row4 = self.table.get([4])

        rows = [row0, row1, row4]
        rows = np.concatenate(rows)
        np.testing.assert_array_equal(rows, values)

    def test_create_embedding_table(self):
        embedding_pb = EmbeddingTableInfo()
        embedding_pb.name = self.name
        embedding_pb.dim = self.dim
        embedding_pb.initializer = self.initializer
        table = create_embedding_table(embedding_pb)
        self.assertIsNotNone(table)
        self.assertEqual(table.name, self.name)
        self.assertEqual(
            tf.keras.initializers.get(self.initializer).__class__,
            table.initializer.__class__,
        )
        self.assertEqual(table.dim, self.dim)

    def test_create_embedding_table_for_slots(self):
        slot_name = "momentum"
        init_value = 3.5
        table = EmbeddingTable(
            get_slot_table_name(self.name, slot_name),
            dim=self.dim,
            initializer=init_value,
            is_slot=True,
        )
        self.assertIsNotNone(table)
        self.assertEqual(table.name, get_slot_table_name(self.name, slot_name))
        self.assertEqual(table.dim, self.dim)
        # test initialize
        embedding = table.get([2])
        self.assertTrue((embedding - init_value < 0.0001).all())


if __name__ == "__main__":
    unittest.main()
