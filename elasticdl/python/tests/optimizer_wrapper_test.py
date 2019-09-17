import copy
import os
import random
import unittest

import mock
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import (
    SGD,
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    Ftrl,
    Nadam,
    RMSprop,
)
from tensorflow.python.ops import init_ops

from elasticdl.python.common.model_helper import (
    find_layer,
    get_module_file_path,
    get_non_embedding_trainable_vars,
    load_module,
)
from elasticdl.python.elasticdl.layers.embedding import Embedding
from elasticdl.python.master.embedding_service import EmbeddingService
from elasticdl.python.master.optimizer_wrapper import (
    OptimizerWrapper,
    _parse_lookup_values,
)
from elasticdl.python.tests.mock_kv_store import MockKvStore


def _prepare_random_data(
    iters_per_epoch,
    batch_size,
    input_length,
    input_dim,
    is_sparse,
    random_seed,
):
    """
    Generate data for training. `is_sparse=False` means we require that all
    embedding ids appear in every batch data.
    """
    random.seed(random_seed)
    X, Y = [], []
    if not is_sparse:
        assert input_length > input_dim, "`input_length` should be larger "
        "than `input_dim` when dense data required."

    def _gen_single_data(choices, input_length, is_sparse):
        if not is_sparse:
            data = copy.copy(choices)
        else:
            data = []
        while len(data) < input_length:
            data.append(random.choice(choices))
        random.shuffle(data)
        return data

    for i in range(iters_per_epoch):
        f1_batch, f2_batch, f3_batch, y_batch = [], [], [], []
        if is_sparse:
            choices = random.choices(range(input_dim), k=input_dim // 2)
        else:
            choices = list(range(input_dim))
        for j in range(batch_size):
            f1_batch.append(_gen_single_data(choices, input_length, is_sparse))
            f2_batch.append(_gen_single_data(choices, input_length, is_sparse))
            f3_batch.append(_gen_single_data(choices, input_length, is_sparse))
            y_batch.append(random.randint(0, 1))
        X.append(
            {
                "f1": np.array(f1_batch),
                "f2": np.array(f2_batch),
                "f3": np.array(f3_batch),
            }
        )
        Y.append(y_batch)
    return X, Y


def _train(model, optimizer, X, Y, loss_fn, random_seed):
    """Train model with TensorFlow optimizer."""
    tf.random.set_seed(random_seed)
    for features, labels in zip(X, Y):
        with tf.GradientTape() as tape:
            outputs = model.call(features)
            loss = loss_fn(outputs, labels)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


def _train_edl_embedding_with_optimizer_wrapper(
    model, opt_keras, X, Y, loss_fn, embed_dims, random_seed
):
    """Train model with optimizer wrapper."""
    tf.random.set_seed(random_seed)
    optimizer = OptimizerWrapper(opt_keras, None, embed_dims)

    # initialization process related to embedding layer and optimizer wrapper
    embed_layers = find_layer(model, Embedding)

    # training process
    for train_iter, (features, labels) in enumerate(zip(X, Y)):
        with tf.GradientTape() as tape:
            for layer in embed_layers:
                layer.set_tape(tape)
            outputs = model.call(features)
            loss = loss_fn(outputs, labels)

        # Need to get non-embedding variables inside for loop because model
        # creates variables after the first time `model.call` is called
        if not train_iter:
            non_embed_vars = get_non_embedding_trainable_vars(
                model, embed_layers
            )
        embed_items = []
        for layer in embed_layers:
            embed_items.extend(
                [
                    (bet, layer.name, ids)
                    for bet, ids in layer.embedding_and_ids
                ]
            )

        grads = tape.gradient(
            loss, non_embed_vars + [var for var, _, _ in embed_items]
        )

        # TODO: do not need to merge gradient from the same embedding layer
        # after `optimizer_wrapper` support grads_and_vars with duplicated
        # layer name
        non_embed_vars_n = len(non_embed_vars)
        non_embed_grads = grads[:non_embed_vars_n]
        embed_grads_dict = {}
        for (_, layer_name, ids), grad in zip(
            embed_items, grads[non_embed_vars_n:]
        ):
            if layer_name in embed_grads_dict:
                merged_grads = embed_grads_dict[layer_name]
                embed_grads_dict[layer_name] = tf.IndexedSlices(
                    tf.concat([merged_grads.values, grad.values], axis=0),
                    tf.concat([merged_grads.indices, ids], axis=0),
                )
            else:
                embed_grads_dict[layer_name] = tf.IndexedSlices(
                    grad.values, ids
                )

        optimizer.apply_gradients(
            list(zip(non_embed_grads, non_embed_vars))
            + [
                (grad, layer_name)
                for layer_name, grad in embed_grads_dict.items()
            ]
        )

        for layer in embed_layers:
            layer.reset()


class OptimizerWrapperTest(unittest.TestCase):
    def _compare_slot_names(self, opt, expected):
        tmp = OptimizerWrapper(opt, None, {})
        self.assertTrue(sorted(tmp.allowed_slot_names) == sorted(expected))

    def test_allowed_slot_names(self):
        opt_and_slots_pairs = [
            (SGD(), []),
            (SGD(momentum=0.2), ["momentum"]),
            (Adam(), ["m", "v"]),
            (Adam(amsgrad=True), ["m", "v", "vhat"]),
            (Adamax(), ["m", "v"]),
            (Nadam(), ["m", "v"]),
            (Adadelta(), ["accum_grad", "accum_var"]),
            (Adagrad(), ["accumulator"]),
            (Ftrl(), ["accumulator", "linear"]),
            (RMSprop(), ["rms"]),
            (RMSprop(momentum=0.2), ["rms", "momentum"]),
            (RMSprop(centered=True), ["rms", "mg"]),
            (RMSprop(momentum=0.2, centered=True), ["rms", "momentum", "mg"]),
        ]
        for opt, expected_slots in opt_and_slots_pairs:
            self._compare_slot_names(opt, expected_slots)

    def _compare_initialize_values(self, opt, dim, slot, expected_init):
        tmp = OptimizerWrapper(opt, None, {"test": dim})
        self.assertTrue(
            (
                tmp._initialize_unknown_slot("test", slot)
                - expected_init(dim).numpy()
                < 0.0001
            ).all()
        )

    def test_initialize(self):
        self._compare_initialize_values(
            Adam(), 4, "m", init_ops.constant_initializer(0.0)
        )
        self._compare_initialize_values(
            Ftrl(initial_accumulator_value=0.5),
            4,
            "accumulator",
            init_ops.constant_initializer(0.5),
        )
        self._compare_initialize_values(
            Adagrad(initial_accumulator_value=0.5),
            4,
            "accumulator",
            init_ops.constant_initializer(0.5),
        )

    def test_initialize_in_lookup(self):
        opt = Adam()
        opt_wrapper = OptimizerWrapper(opt, None, {"test_1": 4})
        grads_and_vars = [(tf.IndexedSlices(None, tf.constant([0])), "test_1")]
        mock_kv_store = MockKvStore({})
        mock_kv_store.update(
            [Embedding.get_key(["test_1", 0])],
            [np.random.rand(4).astype(np.float32)],
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
                    [Embedding.get_key([layer, id])],
                    [np.random.rand(embedding_dim).astype(np.float32)],
                )
                for i, slot in enumerate(["m", "v"]):
                    mock_kv_store.update(
                        [Embedding.get_key([layer, slot, id])],
                        [np.random.rand(embedding_dim).astype(np.float32)],
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
                [Embedding.get_key([layer, id]) for id in ids]
            )
            values = np.concatenate(values).reshape(-1, embedding_dim)
            self.assertTrue((embeddings[layer] - values < 0.0001).all())

            for slot in ["m", "v"]:
                values, _ = mock_kv_store.lookup(
                    [Embedding.get_key([layer, slot, id]) for id in ids]
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
                ).all()
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
            ["test_1-1", "test_1-5", "test_2-10"], [t, t * 5.0, t * 10.0]
        )
        expected_mock_kv_store.update(
            ["test_1-momentum-1", "test_1-momentum-5", "test_2-momentum-10"],
            [t / 10.0, t / 2.0, t],
        )
        for k, ids in zip(["test_1", "test_2"], ids_list):
            for id in ids:
                key = Embedding.get_key([k, id])
                v, _ = mock_kv_store.lookup([key])
                expected_v, _ = expected_mock_kv_store.lookup([key])
                self.assertTrue((v[0] == expected_v[0]).all())

    def _random_init_model_weight(self, shapes, random_seed):
        np.random.seed(random_seed)
        return [np.random.rand(*shape).astype(np.float32) for shape in shapes]

    def _test_correctness(self, optimizer_class, X, Y, seed, **kwargs):
        """Test the correctness of specific TensorFlow optimizer."""
        _model_file = get_module_file_path(
            os.path.dirname(os.path.realpath(__file__)),
            "embedding_test_module.KerasEmbeddingModel",
        )
        model_module = load_module(_model_file).__dict__

        # train model with TensorFlow optimizer
        weights = self._random_init_model_weight(
            [(4, 4), (4, 4), (72, 1), (1,)], seed
        )
        loss_fn = model_module["loss"]
        model1 = model_module["KerasEmbeddingModel"](4, 4, weights)
        opt1 = optimizer_class(**kwargs)
        _train(model1, opt1, X, Y, loss_fn, random_seed=seed)

        model2 = model_module["EdlEmbeddingModel"](4, weights[2:])
        opt2 = optimizer_class(**kwargs)

        layer_names = [layer.name for layer in find_layer(model2, Embedding)]
        embed_dims = dict([(layer_name, 4) for layer_name in layer_names])

        # intialize embedding vectors in kv store
        mock_kv_store = MockKvStore({})
        for layer, embed_table in zip(layer_names, weights[:2]):
            for i, embed_vector in enumerate(embed_table):
                mock_kv_store.update(["%s-%d" % (layer, i)], [embed_vector])

        # train model with optimizer wrapper
        with mock.patch.object(
            EmbeddingService, "lookup_embedding", mock_kv_store.lookup
        ), mock.patch.object(
            EmbeddingService, "update_embedding", mock_kv_store.update
        ):
            _train_edl_embedding_with_optimizer_wrapper(
                model2, opt2, X, Y, loss_fn, embed_dims, random_seed=seed
            )

        # compare trained parameters
        wrong_msg = (
            "The updated parameters of Optimizer Wrapper and TensorFlow "
            "optimizer %s differ." % opt1.get_config()["name"]
        )

        for layer1, layer2 in zip(model1.layers, model2.layers):
            if "embedding" in layer2.name:
                w1 = layer1.weights[0].numpy()
                keys = [Embedding.get_key([layer2.name, i]) for i in range(4)]
                w2 = np.concatenate(mock_kv_store.lookup(keys)[0]).reshape(
                    4, -1
                )
                self.assertTrue((w1 - w2 < 0.0001).all(), msg=wrong_msg)
            else:
                for w1, w2 in zip(layer1.weights, layer2.weights):
                    self.assertTrue(
                        (w1 - w2 < 0.0001).numpy().all(), msg=wrong_msg
                    )

    def test_correctness(self):
        """
        Test the correctness of Optimizer Wrapper for all TensorFlow
        optimizers.
        """
        optimizer_kargs = {
            SGD: {"momentum": 0.5},
            Adadelta: {},
            Adagrad: {},
            Adamax: {},
            Ftrl: {},
            Adam: {"amsgrad": True},
            Nadam: {},
            RMSprop: {"momentum": 0.5, "centered": True},
        }
        learning_rate = 0.1
        for key in optimizer_kargs.keys():
            optimizer_kargs[key]["learning_rate"] = learning_rate

        # TensorFlow implements these optimizers in densely updating style,
        # i.e. update all parameters even if some parameters do not used in
        # forward pass. `OptimizerWrapper` only supports sparsely updating
        # style. So we test these optimizers using dense data for many
        # iterations and sparse data for one iteration.
        tf_dense_optimizers = [Adam, Nadam, RMSprop]

        seed = 1
        _prepare_data_common_args = {
            "batch_size": 4,
            "input_length": 6,
            "input_dim": 4,
            "random_seed": seed,
        }
        X_sparse, Y_sparse = _prepare_random_data(
            iters_per_epoch=4, is_sparse=True, **_prepare_data_common_args
        )
        X_sparse_one_iter, Y_sparse_one_iter = _prepare_random_data(
            iters_per_epoch=1, is_sparse=True, **_prepare_data_common_args
        )
        X_dense, Y_dense = _prepare_random_data(
            iters_per_epoch=4, is_sparse=False, **_prepare_data_common_args
        )

        for opt, kargs in optimizer_kargs.items():
            if opt not in tf_dense_optimizers:
                self._test_correctness(opt, X_sparse, Y_sparse, seed, **kargs)
            else:
                self._test_correctness(opt, X_dense, Y_dense, seed, **kargs)
                self._test_correctness(
                    opt, X_sparse_one_iter, Y_sparse_one_iter, seed, **kargs
                )


if __name__ == "__main__":
    unittest.main()
