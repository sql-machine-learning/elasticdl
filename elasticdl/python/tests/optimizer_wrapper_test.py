import unittest

import mock
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers as tf_optimizers

from elasticdl.python.common.embedding_service import EmbeddingService
from elasticdl.python.elasticdl.layers.embedding import Embedding
from elasticdl.python.master.optimizer_wrapper import OptimizerWrapper
from elasticdl.python.tests.mock_kv_store import MockKvStore


class OptimizerWrapperTest(unittest.TestCase):
    def _compare_slot_names(self, opt, expected):
        tmp = OptimizerWrapper(opt, None, {})
        self.assertTrue(sorted(tmp.allowed_slot_names) == sorted(expected))

    def test_allowed_slot_names(self):
        self._compare_slot_names(tf.keras.optimizers.SGD(), [])
        self._compare_slot_names(
            tf.keras.optimizers.SGD(momentum=0.2), ["momentum"]
        )
        self._compare_slot_names(tf.keras.optimizers.Adam(), ["m", "v"])
        self._compare_slot_names(
            tf.keras.optimizers.Adam(amsgrad=True), ["m", "v", "vhat"]
        )

    def _compare_initialize_values(self, opt, dim, expected_init):
        tmp = OptimizerWrapper(opt, None, {"test": dim})
        self.assertTrue(
            (
                tmp._initialize_unknown_slot("test") - expected_init(dim)
                < 0.0001
            ).all()
        )

    def test_initialize(self):
        self._compare_initialize_values(
            tf.keras.optimizers.Adam(), 4, np.zeros
        )

    def test_initialize_in_lookup(self):
        opt = tf.keras.optimizers.Adam()
        opt_wrapper = OptimizerWrapper(opt, None, {"test-1": 4})
        grads_and_vars = [(tf.IndexedSlices(None, tf.constant([0])), "test-1")]
        mock_kv_store = MockKvStore({})
        mock_kv_store.update(
            keys=[Embedding.get_key(["test-1", 0])],
            values=[np.random.rand(4).astype(np.float32)],
        )
        with mock.patch.object(
            EmbeddingService, "lookup_embedding", mock_kv_store.lookup
        ):
            embeddings, slot_values = opt_wrapper._lookup_embeddings_and_slots(
                grads_and_vars
            )
        self.assertTrue((slot_values["test-1"]["m"] < 0.0001).all())
        self.assertTrue((slot_values["test-1"]["v"] < 0.0001).all())

    def test_lookup(self):
        opt = tf.keras.optimizers.Adam()
        opt_wrapper = OptimizerWrapper(opt, None, {})
        embedding_dim = 4
        layers = ["embedding_0", "embedding-1"]
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
        all_values = np.arange(48).reshape(12, 4)

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

        opt = tf_optimizers.Adam()
        opt_wrapper = OptimizerWrapper(opt, None, embedding_dims)
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
                    slots_dict[slot]
                    == opt_wrapper._slot_variables[layer][slot]
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

        opt = tf_optimizers.SGD()
        opt_wrapper = OptimizerWrapper(opt, None, embedding_dims)
        grads_and_vars = [
            ("test-1-grads", "test-1"),
            ("test-2-grads", "test-2"),
        ]
        opt_wrapper._set_embedding_values_to_variables(
            grads_and_vars, embedding_values
        )
        for i, layer in enumerate(layers):
            self.assertTrue(
                (
                    opt_wrapper._embedding_variables[layer].numpy()
                    == embedding_values[layer]
                ).all()
            )
            self.assertTrue(
                grads_and_vars[i][1] == opt_wrapper._embedding_variables[layer]
            )

        embedding_values_new = {"test-1": np.zeros((3, 4), np.float32)}
        grads_and_vars = [("test-1-grads", "test-1")]
        opt_wrapper._set_embedding_values_to_variables(
            grads_and_vars, embedding_values_new
        )
        self.assertTrue(
            (opt_wrapper._embedding_variables["test-1"].numpy() < 0.0001).all()
        )


if __name__ == "__main__":
    unittest.main()
