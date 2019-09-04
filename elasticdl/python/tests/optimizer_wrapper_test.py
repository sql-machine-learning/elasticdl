import unittest

import mock
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam

from elasticdl.python.common.embedding_service import EmbeddingService
from elasticdl.python.elasticdl.layers.embedding import Embedding
from elasticdl.python.master.optimizer_wrapper import (
    OptimizerWrapper,
    _parse_lookup_values,
)
from elasticdl.python.tests.mock_kv_store import MockKvStore


class OptimizerWrapperTest(unittest.TestCase):
    def _compare_slot_names(self, opt, expected):
        tmp = OptimizerWrapper(opt, None, {})
        self.assertTrue(sorted(tmp.allowed_slot_names) == sorted(expected))

    def test_allowed_slot_names(self):
        self._compare_slot_names(SGD(), [])
        self._compare_slot_names(SGD(momentum=0.2), ["momentum"])
        self._compare_slot_names(Adam(), ["m", "v"])
        self._compare_slot_names(Adam(amsgrad=True), ["m", "v", "vhat"])

    def _compare_initialize_values(self, opt, dim, slot, expected_init):
        tmp = OptimizerWrapper(opt, None, {"test": dim})
        self.assertTrue(
            (
                tmp._initialize_unknown_slot("test", slot) - expected_init(dim)
                < 0.0001
            ).all()
        )

    def test_initialize(self):
        self._compare_initialize_values(Adam(), 4, "m", np.zeros)

    def test_initialize_in_lookup(self):
        opt = Adam()
        opt_wrapper = OptimizerWrapper(opt, None, {"test_1": 4})
        grads_and_vars = [(tf.IndexedSlices(None, tf.constant([0])), "test_1")]
        mock_kv_store = MockKvStore({})
        mock_kv_store.update(
            keys=[Embedding.get_key(["test_1", 0])],
            values=[np.random.rand(4).astype(np.float32)],
        )
        with mock.patch.object(
            EmbeddingService, "lookup_embedding", mock_kv_store.lookup
        ):
            embeddings, slot_values = opt_wrapper._lookup_embeddings_and_slots(
                grads_and_vars
            )
        self.assertTrue((slot_values["test_1"]["m"] < 0.0001).all())
        self.assertTrue((slot_values["test_1"]["v"] < 0.0001).all())

    def test_generate_lookup_keys(self):
        opt = Adam(amsgrad=True)
        opt_wrapper = OptimizerWrapper(opt, None, {})
        slots = ["m", "v", "vhat"]
        layers = ["test_0", "test_1"]
        grads = [
            tf.IndexedSlices(None, tf.constant([2, 0, 2])),
            tf.IndexedSlices(None, tf.constant([1, 2, 0, 2])),
        ]
        ids_list = [[2, 0], [1, 2, 0]]
        grads_and_vars = list(zip(grads, layers))
        arr = opt_wrapper._generate_lookup_keys(grads_and_vars)
        embed_keys, slot_keys, embed_layer_index, slot_layer_index = arr

        expected_embed_keys = [
            Embedding.get_key([layer, id])
            for layer, ids in zip(layers, ids_list)
            for id in ids
        ]
        self.assertTrue(embed_keys == expected_embed_keys)
        expected_slot_keys = [
            Embedding.get_key([layer, slot, id])
            for layer, ids in zip(layers, ids_list)
            for slot in slots
            for id in ids
        ]
        self.assertTrue(slot_keys == expected_slot_keys)

        expected_embed_layer_index = {"test_0": (0, 2), "test_1": (2, 5)}
        self.assertTrue(embed_layer_index == expected_embed_layer_index)
        expected_slot_layer_index = {
            "test_0": {"m": (0, 2), "v": (2, 4), "vhat": (4, 6)},
            "test_1": {"m": (6, 9), "v": (9, 12), "vhat": (12, 15)},
        }
        self.assertTrue(slot_layer_index == expected_slot_layer_index)

        for layer, ids in zip(layers, ids_list):
            self.assertTrue(
                (opt_wrapper._unique_ids_all_layers[layer] == ids).all()
            )

    def test_parse_lookup_values(self):
        dim = 4
        embed_table = [np.random.rand(dim) for i in range(20)]
        key_index = {
            "test_0": {"m": (3, 10), "v": (11, 20)},
            "test_1": {"m": (0, 3), "v": (10, 11)},
        }
        values = _parse_lookup_values(embed_table, key_index)
        expected_values = {
            "test_0": {
                "m": np.concatenate(embed_table[3:10]).reshape(7, dim),
                "v": np.concatenate(embed_table[11:20]).reshape(9, dim),
            },
            "test_1": {
                "m": np.concatenate(embed_table[0:3]).reshape(3, dim),
                "v": np.concatenate(embed_table[10:11]).reshape(1, dim),
            },
        }
        for layer in ["test_0", "test_1"]:
            for slot in ["m", "v"]:
                self.assertTrue(
                    (values[layer][slot] == expected_values[layer][slot]).all()
                )

    def test_lookup(self):
        opt = Adam()
        opt_wrapper = OptimizerWrapper(opt, None, {})
        embedding_dim = 4
        layers = ["embedding_0", "embedding_1"]
        grads = [
            tf.IndexedSlices(None, tf.constant([2, 0, 2])),
            tf.IndexedSlices(None, tf.constant([1, 2, 0, 2])),
        ]
        ids_list = [[2, 0], [1, 2, 0]]
        grads_and_vars = list(zip(grads, layers))
        mock_kv_store = MockKvStore({})
        for layer in layers:
            for id in range(3):
                mock_kv_store.update(
                    keys=[Embedding.get_key([layer, id])],
                    values=[np.random.rand(embedding_dim).astype(np.float32)],
                )
                for i, slot in enumerate(["m", "v"]):
                    mock_kv_store.update(
                        keys=[Embedding.get_key([layer, slot, id])],
                        values=[
                            np.random.rand(embedding_dim).astype(np.float32)
                        ],
                    )

        with mock.patch.object(
            EmbeddingService, "lookup_embedding", mock_kv_store.lookup
        ):
            embeddings, slot_values = opt_wrapper._lookup_embeddings_and_slots(
                grads_and_vars
            )

        grad0 = grads_and_vars[0][0]
        self.assertTrue((grad0.indices.numpy() == [0, 1, 0]).all())
        grad1 = grads_and_vars[1][0]
        self.assertTrue((grad1.indices.numpy() == [0, 1, 2, 1]).all())

        for ids, layer in zip(ids_list, layers):
            self.assertTrue(
                (opt_wrapper._unique_ids_all_layers[layer] == ids).all()
            )

            values, _ = mock_kv_store.lookup(
                keys=[Embedding.get_key([layer, id]) for id in ids]
            )
            values = np.concatenate(values).reshape(-1, embedding_dim)
            self.assertTrue((embeddings[layer] - values < 0.0001).all())

            for slot in ["m", "v"]:
                values, _ = mock_kv_store.lookup(
                    keys=[Embedding.get_key([layer, slot, id]) for id in ids]
                )
                values = np.concatenate(values).reshape(-1, embedding_dim)
                self.assertTrue(
                    (slot_values[layer][slot] - values < 0.0001).all()
                )

    def test_set_slot_values_to_variables(self):
        layers = ["test-1", "test-2"]
        slots = ["m", "v"]
        id_num = 3
        embedding_dims = {layer: 4 for layer in layers}
        all_values = np.arange(48).reshape(12, 4).astype(np.float32)

        slot_values = {}
        offset = 0
        for layer in layers:
            for slot in slots:
                start = offset
                end = offset + id_num
                slot_values.setdefault(layer, {}).setdefault(
                    slot, all_values[start:end]
                )
                offset = end

        opt = Adam()
        opt_wrapper = OptimizerWrapper(opt, None, embedding_dims)
        for layer in layers:
            opt_wrapper._create_embedding_variable(layer, tf.zeros((1, 4)))
        opt_wrapper._set_slot_values_to_variables(slot_values)
        self.assertTrue(len(opt.weights) == 4)
        for layer in layers:
            slots_dict = None
            for k, v in opt._slots.items():
                if k.startswith(layer):
                    slots_dict = v
                    break

            for slot in slots:
                self.assertTrue(
                    (
                        slots_dict[slot].numpy() == slot_values[layer][slot]
                    ).all()
                )
                self.assertTrue(
                    (
                        slots_dict[slot].numpy()
                        == opt_wrapper._slot_variables[layer][slot].numpy()
                    ).all()
                )

                slots_dict[slot].assign(tf.ones((10, 4)))
                self.assertTrue(
                    (
                        opt_wrapper._slot_variables[layer][slot].numpy() - 1.0
                        < 0.0001
                    ).all()
                )
                opt_wrapper._slot_variables[layer][slot].assign(
                    -tf.ones((10, 4))
                )
                self.assertTrue(
                    (slots_dict[slot].numpy() + 1.0 < 0.0001).all()
                )

        slot_values_new = {"test-1": {"m": np.zeros((3, 4), np.float32)}}
        opt_wrapper._set_slot_values_to_variables(slot_values_new)
        self.assertTrue(
            (opt_wrapper._slot_variables["test-1"]["m"].numpy() < 0.0001).all()
        )

    def test_set_embedding_values_to_variables(self):
        layers = ["test-1", "test-2"]
        id_num = 3
        embedding_dims = {layer: 4 for layer in layers}
        all_values = np.arange(16).reshape(4, 4)

        embedding_values = {}
        offset = 0
        for layer in layers:
            start = offset
            end = offset + id_num
            embedding_values.setdefault(layer, all_values[start:end])
            offset = end

        opt = SGD()
        opt_wrapper = OptimizerWrapper(opt, None, embedding_dims)
        grads_and_vars = [
            ("test-1-grads", "test-1"),
            ("test-2-grads", "test-2"),
        ]
        opt_wrapper._set_embedding_values_to_variables(
            grads_and_vars, embedding_values
        )
        for i, layer in enumerate(layers):
            print(opt_wrapper._embed_variables[layer])
            self.assertTrue(
                (
                    opt_wrapper._embed_variables[layer].numpy()
                    == embedding_values[layer]
                ).all()
            )
            self.assertTrue(
                (
                    grads_and_vars[i][1].numpy()
                    == opt_wrapper._embed_variables[layer].numpy()
                ).any()
            )

        embedding_values_new = {"test-1": np.zeros((3, 4), np.float32)}
        grads_and_vars = [("test-1-grads", "test-1")]
        opt_wrapper._set_embedding_values_to_variables(
            grads_and_vars, embedding_values_new
        )
        self.assertTrue(
            (opt_wrapper._embed_variables["test-1"].numpy() < 0.0001).all()
        )

    def test_report_to_kv_store(self):
        opt = SGD(momentum=0.1)
        opt_wrapper = OptimizerWrapper(opt, None, {})

        ids_list = [[1, 5], [10]]
        opt_wrapper._unique_ids_all_layers = {
            "test_1": np.array(ids_list[0]),
            "test_2": np.array(ids_list[1]),
        }
        t = np.array([1.0, 1.0, 1.0])
        opt_wrapper._embed_variables = {
            "test_1": tf.Variable([t, t * 5]),
            "test_2": tf.Variable([t * 10]),
        }
        opt_wrapper._slot_variables = {
            "test_1": {"momentum": tf.Variable([t / 10.0, t / 2.0])},
            "test_2": {"momentum": tf.Variable([t])},
        }

        mock_kv_store = MockKvStore({})
        with mock.patch.object(
            EmbeddingService, "update_embedding", mock_kv_store.update
        ):
            opt_wrapper._report_to_kv_store()

        expected_mock_kv_store = MockKvStore({})
        expected_mock_kv_store.update(
            keys=["test_1-1", "test_1-5", "test_2-10"],
            values=[t, t * 5.0, t * 10.0],
        )
        expected_mock_kv_store.update(
            keys=[
                "test_1-momentum-1",
                "test_1-momentum-5",
                "test_2-momentum-10",
            ],
            values=[t / 10.0, t / 2.0, t],
        )
        for k, ids in zip(["test_1", "test_2"], ids_list):
            for id in ids:
                key = Embedding.get_key([k, id])
                v, _ = mock_kv_store.lookup([key])
                expected_v, _ = expected_mock_kv_store.lookup([key])
                self.assertTrue((v[0] == expected_v[0]).all())


if __name__ == "__main__":
    unittest.main()
