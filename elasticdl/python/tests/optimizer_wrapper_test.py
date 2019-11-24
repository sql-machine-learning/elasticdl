import copy
import os
import random
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

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

from elasticdl.python.common.model_utils import (
    find_layer,
    get_module_file_path,
    get_non_embedding_trainable_vars,
    load_module,
)
from elasticdl.python.elasticdl.layers.embedding import Embedding
from elasticdl.python.master.optimizer_wrapper import OptimizerWrapper
from elasticdl.python.ps.embedding_table import (
    EmbeddingTable,
    get_slot_table_name,
)
from elasticdl.python.ps.parameters import Parameters


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
    model, opt_keras, X, Y, loss_fn, params, random_seed
):
    """Train model with optimizer wrapper."""
    tf.random.set_seed(random_seed)
    opt_wrapper = OptimizerWrapper(
        opt_keras,
        lookup_embedding_func=params.get_embedding_param,
        update_embedding_func=params.set_embedding_param,
    )

    embed_layers = find_layer(model, Embedding)

    # initialize slot params
    params.create_slot_params(
        opt_wrapper.allowed_slot_names, opt_wrapper.slot_initial_value
    )

    # initialize ElasticDL embedding layer
    for layer in embed_layers:
        layer.set_lookup_embedding_func(params.get_embedding_param)

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

        opt_wrapper.apply_gradients(
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
        tmp = OptimizerWrapper(opt)
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

    def test_set_slot_to_optimizer(self):
        embed_name = "test_emb"
        indices = np.ndarray([2], dtype=np.int32)
        embed_values = np.ndarray([2, 2], dtype=np.float32)
        slot_values = {
            "m": np.ndarray([2, 2], dtype=np.float32),
            "v": np.ndarray([2, 2], dtype=np.float32),
        }
        params = Parameters()
        params.embedding_params[embed_name] = EmbeddingTable(embed_name, 8)
        for slot in ["m", "v"]:
            slot_table_name = get_slot_table_name(embed_name, slot)
            params.embedding_params[slot_table_name] = EmbeddingTable(
                slot_table_name, 2, "0.0", True
            )

        opt = Adam()
        opt_wrapper = OptimizerWrapper(opt, None, params.get_embedding_param)
        opt_wrapper._init_thread_local()

        opt_wrapper._tls._unique_ids_all_layers[embed_name] = indices
        opt_wrapper._create_embedding_variable(embed_name, embed_values)
        opt_wrapper._get_slot_and_set_to_optimizer(embed_name)

        self.assertEqual(len(opt._slots), 1)
        opt_slots = list(opt._slots.values())[0]
        self.assertEqual(sorted(opt_slots.keys()), ["m", "v"])
        for name in ["m", "v"]:
            self.assertTrue(
                np.allclose(opt_slots[name].numpy(), slot_values[name])
            )

    def test_report_to_kv_store(self):
        params = Parameters()
        for name in ["test_1", "test_2"]:
            params.embedding_params[name] = EmbeddingTable(name, 8)
            slot_key = get_slot_table_name(name, "momentum")
            params.embedding_params[slot_key] = EmbeddingTable(
                slot_key, 8, "0.0", True
            )

        indices = {
            "test_1": np.array([1, 5]),
            "test_2": np.array([10]),
        }
        embed_vars = {
            "test_1": tf.Variable(np.random.rand(2, 8).astype(np.float32)),
            "test_2": tf.Variable(np.random.rand(1, 8).astype(np.float32)),
        }
        slot_vars = {
            "test_1": {
                "momentum": tf.Variable(
                    np.random.rand(2, 8).astype(np.float32)
                )
            },
            "test_2": {
                "momentum": tf.Variable(
                    np.random.rand(1, 8).astype(np.float32)
                )
            },
        }

        opt = SGD(momentum=0.1)
        opt_wrapper = OptimizerWrapper(
            opt, None, None, params.set_embedding_param
        )
        opt_wrapper._tls._unique_ids_all_layers = indices
        opt_wrapper._tls._embed_variables = embed_vars
        opt_wrapper._tls._slot_variables = slot_vars
        opt_wrapper._report_to_kv_store()

        for name in ["test_1", "test_2"]:
            self.assertTrue(
                np.allclose(
                    embed_vars[name].numpy(),
                    params.get_embedding_param(name, indices[name]),
                )
            )

            slot = "momentum"
            slot_table_name = get_slot_table_name(name, slot)
            self.assertTrue(
                np.allclose(
                    slot_vars[name][slot].numpy(),
                    params.get_embedding_param(slot_table_name, indices[name]),
                )
            )

    def test_delete_variables(self):
        params = Parameters()
        embed_layers = ["test_1", "test_2"]
        slot_names = ["m", "v"]
        dim = 8
        for layer in embed_layers:
            params.embedding_params[layer] = EmbeddingTable(layer, dim)
            for slot in slot_names:
                slot_key = get_slot_table_name(layer, slot)
                params.embedding_params[slot_key] = EmbeddingTable(
                    slot_key, dim, "0.0", True
                )

        opt = Adam()
        opt_wrapper = OptimizerWrapper(
            opt, None, params.get_embedding_param, params.set_embedding_param
        )

        opt_wrapper._init_thread_local()
        for name in embed_layers:
            opt_wrapper._tls._unique_ids_all_layers[name] = np.ndarray(
                [2], np.int32
            )
            opt_wrapper._create_embedding_variable(
                name, np.ndarray([2, dim], np.float32)
            )
            opt_wrapper._get_slot_and_set_to_optimizer(name)

        self.assertTrue(len(opt._weights) == 4)
        self.assertTrue(len(opt._slots) == 2)
        for slot_dict in opt._slots.values():
            self.assertTrue(len(slot_dict) == 2)

        opt_wrapper._delete_variables()
        self.assertTrue(len(opt._weights) == 0)
        self.assertTrue(len(opt._slots) == 0)

    def _random_init_model_weight(self, shapes, random_seed):
        np.random.seed(random_seed)
        return [np.random.rand(*shape).astype(np.float32) for shape in shapes]

    def _test_correctness(self, optimizer_class, X, Y, seed, **opt_kwargs):
        """Test the correctness of specific TensorFlow optimizer."""
        _model_file = get_module_file_path(
            os.path.dirname(os.path.realpath(__file__)),
            "embedding_test_module.KerasEmbeddingModel",
        )
        model_module = load_module(_model_file).__dict__

        # train model with TensorFlow optimizer
        dim = 4
        weights = self._random_init_model_weight(
            [(4, dim), (4, dim), (72, 1), (1,)], seed
        )
        loss_fn = model_module["loss"]
        model1 = model_module["KerasEmbeddingModel"](4, dim, weights)
        opt1 = optimizer_class(**opt_kwargs)
        _train(model1, opt1, X, Y, loss_fn, random_seed=seed)

        model2 = model_module["EdlEmbeddingModel"](dim, weights[2:])
        opt2 = optimizer_class(**opt_kwargs)

        layer_names = [layer.name for layer in find_layer(model2, Embedding)]

        # create Parameters object and initialize embedding vectors
        params = Parameters()
        for layer_name, embed_value in zip(layer_names, weights[:2]):
            embed_table = EmbeddingTable(layer_name, dim)
            embed_table.set(range(len(embed_value)), embed_value)
            params.embedding_params[layer_name] = embed_table

        _train_edl_embedding_with_optimizer_wrapper(
            model2, opt2, X, Y, loss_fn, params, random_seed=seed
        )

        # compare trained parameters
        wrong_msg = (
            "The updated parameters of Optimizer Wrapper and TensorFlow "
            "optimizer %s differ." % opt1.get_config()["name"]
        )

        for layer1, layer2 in zip(model1.layers, model2.layers):
            if "embedding" in layer2.name:
                w1 = layer1.weights[0].numpy()
                w2 = params.get_embedding_param(layer2.name, range(4))
                self.assertTrue(np.isclose(w1, w2).all(), msg=wrong_msg)
            else:
                for w1, w2 in zip(layer1.weights, layer2.weights):
                    self.assertTrue(
                        np.isclose(w1.numpy(), w2.numpy()).all(), msg=wrong_msg
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

    def _test_async_correctness(
        self,
        grads_and_vars_batches,
        embed_values,
        expected_non_embed_values,
        expected_embed_values=None,
    ):
        """Checks the correctness of async OptimizerWrapper. This function
        creates many threads and these threads call
        `OptimizerWrapper.apply_gradients` simultaneously.

        Args:
            grads_and_vars_batches: A python list of `grads_and_vars`. Every
                thread takes a `grads_and_vars` and calls `apply_gradients`.
            embed_values: A python dictionary of
                `(layer_name, embedding table)`.
            expected_non_embed_values: A python list of expected non-embdding
                values after applying gradients.
            expected_embed_values: A python dictionary of expected embedding
                values after applying gradients. None means no need to check
                embedding values.
        """
        thread_num = len(grads_and_vars_batches)
        input_dims = {}
        embed_var_n = len(embed_values)
        params = Parameters()
        for layer, values in embed_values.items():
            embed_dim = values.shape[1]
            input_dims[layer] = values.shape[0]
            embed_table = EmbeddingTable(layer, embed_dim)
            embed_table.set(range(input_dims[layer]), values)
            params.embedding_params[layer] = embed_table

        opt = SGD(0.1)
        opt_wrapper = OptimizerWrapper(
            opt,
            True,
            lookup_embedding_func=params.get_embedding_param,
            update_embedding_func=params.set_embedding_param,
        )

        # call optimizer_wrapper.apply_gradients asynchronously
        def _apply_gradients(opt_wrapper, grads_and_vars):
            # sleep 1s to wait that all threads are in this method call
            time.sleep(1)
            opt_wrapper.apply_gradients(grads_and_vars)

        executor = ThreadPoolExecutor(max_workers=thread_num)
        tasks = [
            executor.submit(_apply_gradients, opt_wrapper, grads_and_vars)
            for grads_and_vars in grads_and_vars_batches
        ]
        _ = [task.result() for task in tasks]

        # check updated results of non-embedding variables
        non_embed_vars = [
            var for grad, var in grads_and_vars_batches[0][:-embed_var_n]
        ]
        for var, expected_value in zip(
            non_embed_vars, expected_non_embed_values
        ):
            self.assertTrue(np.isclose(var.numpy(), expected_value).all())

        # `expected_embed_values=None` means that no need to check
        # embedding table
        if not expected_embed_values:
            return
        # check updated results of embedding table
        for layer, expected_values in expected_embed_values.items():
            value = params.get_embedding_param(layer, range(input_dims[layer]))

            self.assertTrue(
                any(
                    [
                        np.isclose(value, expected).all()
                        for expected in expected_values
                    ]
                )
            )

    def test_async_correctness(self):
        """Tests the correctness of async updates in `OptimizerWrapper`.

        Testing the correctness is not simple because OptimizerWrapper is not
        thread-safe for embedding table. This test case lists all the possible
        results when `thread_number=2` and test the correctness. This test case
        also tests that OptimizerWrapper does not raise Error with a large
        thread number(8).
        """
        max_thread_num = 8
        input_dim = 4
        output_dim = 3
        non_embed_vars = [
            tf.Variable([0.0] * output_dim),
            tf.Variable([1.0] * output_dim),
        ]
        non_embed_vars_copy = copy.deepcopy(non_embed_vars)
        non_embed_grads_batches = [
            [
                tf.constant([i + 1] * output_dim, dtype=tf.float32),
                tf.constant([-i - 1] * output_dim, dtype=tf.float32),
            ]
            for i in range(max_thread_num)
        ]
        embed_shape = (input_dim, output_dim)
        embed_value_count = output_dim * input_dim
        embed_layers = ["embed_1", "embed_2"]
        embed_values = {
            embed_layers[0]: np.arange(
                embed_value_count, dtype=np.float32
            ).reshape(embed_shape),
            embed_layers[1]: np.arange(
                embed_value_count, dtype=np.float32
            ).reshape(embed_shape),
        }

        embed_grads_batches = [
            [
                tf.IndexedSlices(
                    tf.reshape(
                        tf.constant([i + 1.0] * embed_value_count), embed_shape
                    ),
                    tf.constant(list(range(input_dim))),
                ),
                tf.IndexedSlices(
                    tf.reshape(
                        tf.constant([-i - 1.0] * embed_value_count),
                        embed_shape,
                    ),
                    tf.constant(list(range(input_dim))),
                ),
            ]
            for i in range(max_thread_num)
        ]

        # thread number = 2
        expected_non_embed_values = [[-0.3, -0.3, -0.3], [1.3, 1.3, 1.3]]
        expected_embed_values = {
            embed_layers[0]: [
                (np.arange(12) - 0.1).reshape(embed_shape),
                (np.arange(12) - 0.2).reshape(embed_shape),
                (np.arange(12) - 0.3).reshape(embed_shape),
            ],
            embed_layers[1]: [
                (np.arange(12) + 0.1).reshape(embed_shape),
                (np.arange(12) + 0.2).reshape(embed_shape),
                (np.arange(12) + 0.3).reshape(embed_shape),
            ],
        }
        grads_and_vars_batches = [
            list(zip(non_embed_grads_batches[i], non_embed_vars))
            + list(zip(embed_grads_batches[i], embed_layers))
            for i in range(2)
        ]

        self._test_async_correctness(
            grads_and_vars_batches,
            embed_values,
            expected_non_embed_values,
            expected_embed_values,
        )

        # thread number = 8
        grads_sum = max_thread_num * (max_thread_num + 1) / 2 / 10.0
        expected_non_embed_values = [[-grads_sum] * 3, [1 + grads_sum] * 3]
        grads_and_vars_batches = [
            list(zip(non_embed_grads_batches[i], non_embed_vars_copy))
            + list(zip(embed_grads_batches[i], embed_layers))
            for i in range(max_thread_num)
        ]
        # Do not check updating results of embedding table when `thread_num>2`.
        # Because there are too many possible results.
        self._test_async_correctness(
            grads_and_vars_batches, embed_values, expected_non_embed_values
        )


if __name__ == "__main__":
    unittest.main()
