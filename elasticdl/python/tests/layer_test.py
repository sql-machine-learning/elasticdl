import os
import unittest

import numpy as np
import tensorflow as tf

from elasticdl.python.common.model_helper import (
    find_layer,
    get_model_file,
    load_model_from_module,
    load_module,
)
from elasticdl.python.elasticdl.layers.embedding import Embedding


def _get_model_zoo_path():
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../../../model_zoo"
    )


def _create_model_instance(model_def):
    module_file = get_model_file(_get_model_zoo_path(), model_def)
    model_module = load_module(module_file).__dict__
    return load_model_from_module(model_def, model_module, None)


class FindLayerTest(unittest.TestCase):
    def test_find_layer(self):
        model_def = "mnist_functional_api.mnist_functional_api.CustomModel"
        model = _create_model_instance(model_def).get_model()

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

    def embedding_lookup(self, ids, name, embedding_initializer):
        values = np.take(self.embedding, ids, axis=0)
        return values


def create_embedding_layer(embedding_size, output_dim, input_length=None):
    layer = Embedding(output_dim, input_length=input_length)
    worker = mock_worker(embedding_size, output_dim)
    layer.set_worker(worker)
    return layer


@tf.function
def layer_call(layer, inputs):
    return layer.call(inputs)


class EmbeddingLayerTest(unittest.TestCase):
    def test_embedding_layer(self):
        output_dim = 8
        embedding_size = 16
        layer = create_embedding_layer(embedding_size, output_dim)

        input_shape = (2, 6)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, input_shape + (output_dim,))

        ids = [0, 1, 3, 8, 3, 2, 3]
        values = layer.call(ids)
        values = values.numpy()
        for index, idx in enumerate(ids):
            correct_value = np.array([idx] * output_dim, dtype=np.float32)
            self.assertTrue((values[index] == correct_value).all())

        results = layer_call(layer, ids)
        results = results.numpy()
        for index, idx in enumerate(ids):
            self.assertTrue((values[index] == results[index]).all())

        model = tf.keras.models.Sequential([layer])
        outputs = model.call(np.array([ids, ids]))
        values = outputs.numpy()[1]
        for index, idx in enumerate(ids):
            correct_value = np.array([idx] * output_dim, dtype=np.float32)
            self.assertTrue((values[index] == correct_value).all())

    def test_embedding_layer_with_input_length(self):
        output_dim = 8
        embedding_size = 16
        input_length = 4
        layer = create_embedding_layer(
            embedding_size, output_dim, input_length=input_length
        )
        ids = [[0, 1, 3, 8], [5, 3, 2, 3]]
        flatten_ids = ids[0] + ids[1]
        values = layer.call(ids)
        values = values.numpy().reshape(-1, output_dim)
        for index, idx in enumerate(flatten_ids):
            correct_value = np.array([idx] * output_dim, dtype=np.float32)
            self.assertTrue((values[index] == correct_value).all())

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
            with tf.GradientTape() as tape:
                layer.set_tape(tape)
                output = layer.call(inputs)
                output = output * multiply_tensor
            bet = layer.bet_ids_pair[0][0]
            grads = tape.gradient(output, bet)
            layer.reset()
            self.assertTrue((grads.values.numpy() == multiply_values).all())

        for inputs in inputs_list:
            with tf.GradientTape() as tape:
                layer.set_tape(tape)
                self.assertRaises(RuntimeError, layer_call, layer, inputs)


if __name__ == "__main__":
    unittest.main()
