import os
import unittest

import tensorflow as tf

from elasticdl.python.common.model_helper import (
    find_layer,
    get_model_file,
    load_model_from_module,
    load_module,
)


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


if __name__ == "__main__":
    unittest.main()
