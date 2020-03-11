import os
import unittest

import numpy as np
import tensorflow as tf

from elasticdl.python.common.model_utils import (
    find_layer,
    get_module_file_path,
    load_model_from_module,
    load_module,
)
from elasticdl.python.elasticdl.layers.embedding import Embedding


def _get_model_zoo_path():
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../../../model_zoo"
    )


def _create_model_instance(model_def):
    module_file = get_module_file_path(_get_model_zoo_path(), model_def)
    model_module = load_module(module_file).__dict__
    return load_model_from_module(model_def, model_module, None)


class FindLayerTest(unittest.TestCase):
    def test_find_layer(self):
        model_def = "mnist_functional_api.mnist_functional_api.custom_model"
        model = _create_model_instance(model_def)

        layer_num = {
            tf.keras.layers.Conv2D: 2,
            tf.keras.layers.Dropout: 1,
            tf.keras.layers.Embedding: 0,
        }

        for layer_class in layer_num:
            layers = find_layer(model, layer_class)
            self.assertEqual(layer_num[layer_class], len(layers))

    def test_find_layer_nested(self):
        model_def = "resnet50_subclass.resnet50_subclass.CustomModel"
        model = _create_model_instance(model_def)

        layer_num = {
            tf.keras.layers.Conv2D: 53,
            tf.keras.layers.Activation: 50,
            tf.keras.layers.Embedding: 0,
        }

        for layer_class in layer_num:
            layers = find_layer(model, layer_class)
            self.assertEqual(layer_num[layer_class], len(layers))


class mock_worker:
    def __init__(self, embedding_size, output_dim):
        self.embedding_size = embedding_size
        self.output_dim = output_dim
        self.embedding = np.ndarray(
            shape=(embedding_size, output_dim), dtype=np.float32
        )
        for i in range(embedding_size):
            self.embedding[i].fill(i)

    def lookup_embedding(self, name, ids):
        values = np.take(self.embedding, ids, axis=0)
        return values


def create_embedding_layer(
    embedding_size,
    output_dim,
    input_length=None,
    combiner=None,
    mask_zero=False,
):
    layer = Embedding(
        output_dim,
        input_length=input_length,
        combiner=combiner,
        mask_zero=mask_zero,
    )
    worker = mock_worker(embedding_size, output_dim)
    layer.set_lookup_embedding_func(worker.lookup_embedding)
    return layer


def get_correct_values_for_sparse_test(indices, values, dense_shape, combiner):
    results = []
    for i in range(dense_shape[0]):
        embedding_values = []
        for n, idx in enumerate(indices):
            if idx[0] == i:
                embedding_values.append(values[n])
        if combiner == "sum":
            combined_value = np.sum(embedding_values, dtype=np.float32)
        elif combiner == "mean":
            combined_value = np.sum(embedding_values, dtype=np.float32) / len(
                embedding_values
            )
        elif combiner == "sqrtn":
            combined_value = np.sum(
                embedding_values, dtype=np.float32
            ) / np.sqrt(len(embedding_values), dtype=np.float32)
        results.append(combined_value)
    return results


class EmbeddingLayerTest(unittest.TestCase):
    def _run_forward_pass_and_compare(
        self, call_fns, inputs, correct_values, output_dim
    ):
        for call_fn in call_fns:
            values = call_fn(inputs)
            values = values.numpy().reshape(-1, output_dim)
            for v, correct_v in zip(values, correct_values):
                self.assertTrue(np.isclose(v, correct_v).all())

    def test_embedding_layer(self):
        output_dim = 8
        embedding_size = 16
        layer = create_embedding_layer(embedding_size, output_dim)

        input_shape = (2, 6)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, input_shape + (output_dim,))

        ids = [0, 1, 3, 8, 3, 2, 3]
        call_fns = [layer.call, tf.function(layer.call)]
        correct_values = np.array(
            [np.array([idx] * output_dim, dtype=np.float32) for idx in ids]
        )
        self._run_forward_pass_and_compare(
            call_fns, ids, correct_values, output_dim
        )

        # Keras model without/with input_layer
        model_without_input_layer = tf.keras.models.Sequential([layer])
        inputs = tf.keras.Input(shape=(7,))
        embeddings = layer(inputs)
        model_with_input_layer = tf.keras.Model(
            inputs=inputs, outputs=embeddings
        )
        models = [model_without_input_layer, model_with_input_layer]
        for model in models:
            inputs = tf.constant([ids])
            call_fns = [model.call, tf.function(model.call)]
            self._run_forward_pass_and_compare(
                call_fns, inputs, correct_values, output_dim
            )

    def test_embedding_layer_with_input_length(self):
        output_dim = 8
        embedding_size = 16
        input_length = 4
        layer = create_embedding_layer(
            embedding_size, output_dim, input_length=input_length
        )
        ids = [[0, 1, 3, 8], [5, 3, 2, 3]]
        flatten_ids = ids[0] + ids[1]
        call_fns = [layer.call, tf.function(layer.call)]
        correct_values = np.array(
            [
                np.array([idx] * output_dim, dtype=np.float32)
                for idx in flatten_ids
            ]
        )
        self._run_forward_pass_and_compare(
            call_fns, ids, correct_values, output_dim
        )

    def test_embedding_layer_with_sparse_input(self):
        output_dim = 8
        embedding_size = 16
        combiners = ["sum", "mean", "sqrtn"]
        indices = [[0, 0], [1, 1], [1, 2], [2, 1], [3, 1], [3, 3], [3, 5]]
        values = [1, 1, 3, 2, 0, 2, 6]
        dense_shape = [4, 6]

        for combiner in combiners:
            layer = create_embedding_layer(
                embedding_size, output_dim, combiner=combiner
            )
            inputs = tf.SparseTensor(
                indices=indices, values=values, dense_shape=dense_shape
            )
            call_fns = [layer.call, tf.function(layer.call)]
            correct_value_single_line = get_correct_values_for_sparse_test(
                indices, values, dense_shape, combiner
            )
            correct_values = np.array(
                [
                    np.array([idx] * output_dim, dtype=np.float32)
                    for idx in correct_value_single_line
                ]
            )
            self._run_forward_pass_and_compare(
                call_fns, inputs, correct_values, output_dim
            )

    def test_embedding_layer_with_mask_zero(self):
        output_dim = 8
        embedding_size = 16
        mask_zero = True
        layer = create_embedding_layer(
            embedding_size, output_dim, mask_zero=mask_zero
        )
        ids = [[0, 1, 3, 8], [5, 0, 2, 3]]
        correct_masks = np.ones((2, 4), dtype=np.bool)
        correct_masks[0][0] = False
        correct_masks[1][1] = False

        masks = layer.compute_mask(ids)
        self.assertTrue((masks.numpy() == correct_masks).all())

        # SparseTensor inputs will raise error
        indices = [[0, 0], [1, 1], [1, 2], [2, 1], [3, 1], [3, 3], [3, 5]]
        values = [1, 1, 3, 2, 0, 2, 6]
        dense_shape = [4, 6]
        inputs = tf.SparseTensor(
            indices=indices, values=values, dense_shape=dense_shape
        )
        self.assertRaises(ValueError, layer.compute_mask, inputs)

    def test_embedding_layer_gradient(self):
        output_dim = 8
        embedding_size = 16
        layer = create_embedding_layer(embedding_size, output_dim)
        inputs_list = [
            tf.keras.backend.constant([[0, 1, 3], [1, 2, 0]], dtype=tf.int64),
            tf.keras.backend.constant(
                [[0, 10, 3], [11, 2, 1]], dtype=tf.int64
            ),
        ]
        multiply_values = np.ndarray(shape=(6, output_dim), dtype=np.float32)
        for i in range(6):
            multiply_values[i].fill(i)
        multiply_tensor = tf.convert_to_tensor(multiply_values)
        multiply_tensor = tf.reshape(multiply_tensor, [2, 3, 8])

        for inputs in inputs_list:
            call_fns = [layer.call, tf.function(layer.call)]
            for call_fn in call_fns:
                with tf.GradientTape() as tape:
                    layer.set_tape(tape)
                    output = call_fn(inputs)
                    output = output * multiply_tensor
                batch_embedding = layer.embedding_and_ids[0].batch_embedding
                grads = tape.gradient(output, batch_embedding)
                self.assertTrue(
                    (grads.values.numpy() == multiply_values).all()
                )
                self.assertTrue(
                    (
                        layer.embedding_and_ids[0].batch_ids.numpy()
                        == inputs.numpy().reshape(-1)
                    ).all()
                )
                layer.reset()

        # Test when an embedding layer is called more than once.
        # When an embedding layer is called more than once, `tf.function` is
        # not supported.
        correct_ids_list = [[0, 1, 3, 1, 2, 0], [0, 10, 3, 11, 2, 1]]
        with tf.GradientTape() as tape:
            layer.set_tape(tape)
            for inputs in inputs_list:
                _ = layer.call(inputs)
            self.assertTrue(len(layer.embedding_and_ids) == len(inputs_list))
            for i, correct_ids in enumerate(correct_ids_list):
                self.assertTrue(
                    (
                        layer.embedding_and_ids[i].batch_ids.numpy()
                        == correct_ids
                    ).all()
                )

    def test_embedding_layer_gradient_with_sparse_inputs(self):
        output_dim = 8
        embedding_size = 16
        combiners = ["sum", "mean", "sqrtn"]
        indices = [[0, 0], [1, 1], [1, 2], [2, 1], [3, 1], [3, 3], [3, 5]]
        values = [1, 1, 3, 2, 0, 2, 6]
        dense_shape = [4, 6]
        inputs = tf.SparseTensor(
            indices=indices, values=values, dense_shape=dense_shape
        )
        inputs = tf.dtypes.cast(inputs, tf.int64)

        multiply_values = np.ndarray(shape=(4, output_dim), dtype=np.float32)
        for i in range(4):
            multiply_values[i].fill(i)
        multiply_tensor = tf.convert_to_tensor(multiply_values)

        # grads for unique ids (1, 3, 2, 0, 6)
        sum_correct_grads = [0 + 1, 1, 2 + 3, 3, 3]
        mean_correct_grads = [1 / 2.0, 1 / 2.0, 2 + 3 / 3.0, 3 / 3.0, 3 / 3.0]
        sqrtn_correct_grads = [
            1 / np.sqrt(2.0),
            1 / np.sqrt(2.0),
            2 + 3 / np.sqrt(3.0),
            3 / np.sqrt(3.0),
            3 / np.sqrt(3.0),
        ]
        correct_grads = {
            "sum": sum_correct_grads,
            "mean": mean_correct_grads,
            "sqrtn": sqrtn_correct_grads,
        }

        for combiner in combiners:
            layer = create_embedding_layer(
                embedding_size, output_dim, combiner=combiner
            )
            call_fns = [tf.function(layer.call), layer.call]
            for call_fn in call_fns:
                with tf.GradientTape() as tape:
                    layer.set_tape(tape)
                    output = call_fn(inputs)
                    output = output * multiply_tensor
                batch_embedding = layer.embedding_and_ids[0].batch_embedding
                grads = tape.gradient(output, batch_embedding)
                grads = grads.numpy()
                place = 8 if combiner == "sum" else 5
                for n, v in enumerate(correct_grads[combiner]):
                    self.assertAlmostEqual(grads[n][0], v, place)
                self.assertTrue(
                    np.isclose(
                        layer.embedding_and_ids[0].batch_ids.numpy(),
                        np.array([1, 3, 2, 0, 6]),
                    ).all()
                )
                layer.reset()

        # Test when an embedding layer is called more than once.
        # When an embedding layer is called more than once, `tf.function` is
        # not supported.
        correct_ids = [1, 3, 2, 0, 6]
        for combiner in combiners:
            layer = create_embedding_layer(
                embedding_size, output_dim, combiner=combiner
            )
            with tf.GradientTape() as tape:
                layer.set_tape(tape)
                _ = layer.call(inputs)
                _ = layer.call(inputs)
                self.assertTrue(len(layer.embedding_and_ids) == 2)
                for i in range(2):
                    self.assertTrue(
                        (
                            layer.embedding_and_ids[i].batch_ids.numpy()
                            == correct_ids
                        ).all()
                    )

    def test_embedding_layer_with_none_shape_input(self):
        output_dim = 8
        embedding_size = 16
        layer = create_embedding_layer(embedding_size, output_dim)
        input = tf.keras.layers.Input(shape=(None,))
        embedding_output = layer(input)
        embedding_output_shape = embedding_output.get_shape().as_list()
        self.assertEqual(embedding_output_shape, [None, None, 8])


if __name__ == "__main__":
    unittest.main()
