import os
import unittest

import numpy as np
import tensorflow as tf

from elasticdl.python.common.model_handler import ModelHandler
from elasticdl.python.elasticdl.layers.embedding import Embedding
from elasticdl.python.tests.test_module import (
    custom_model_with_embedding,
    custom_sequential_model,
    feature_columns_fn,
)

_model_zoo_path = os.path.dirname(os.path.realpath(__file__))


def _get_dataset():
    y_labels = np.array([1, 1, 0, 0, 1])
    x_data = {
        "age": [14, 56, 78, 38, 80],
        "education": [
            "Bachelors",
            "Master",
            "Some-college",
            "Bachelors",
            "Master",
        ],
    }
    dataset = tf.data.Dataset.from_tensor_slices((dict(x_data), y_labels))
    dataset = dataset.shuffle(len(x_data)).batch(4)
    return dataset


class DefaultModelHandlerTest(unittest.TestCase):
    def setUp(self):
        self.model_handler = ModelHandler.get_model_handler()

    def test_get_model_to_ps(self):
        model_inst = custom_model_with_embedding()
        model_inst = self.model_handler.get_model_to_train(
            model_inst
        )
        self.assertEqual(type(model_inst.layers[1]), tf.keras.layers.Embedding)

    def test_get_model_to_export(self):
        feature_columns = feature_columns_fn()
        model_inst = custom_sequential_model(feature_columns)
        dataset = _get_dataset()
        self.model_handler.get_model_to_export(
            model_inst, dataset
        )
        self.assertEqual(list(model_inst.inputs.keys()), ["age", "education"])
        self.assertEqual(len(model_inst.outputs), 1)


class ParameterSeverModelHandlerTest(unittest.TestCase):
    def setUp(self):
        self.model_handler = ModelHandler.get_model_handler(
            distribution_strategy="ParameterServerStrategy"
        )

    def test_get_model_to_train(self):
        model_inst = custom_model_with_embedding()
        model_inst = self.model_handler.get_model_to_train(
            model_inst
        )
        self.assertEqual(type(model_inst.layers[1]), Embedding)

    def test_get_model_to_export(self):
        pass


if __name__ == "__main__":
    unittest.main()
