import os
import unittest
import tensorflow as tf
import numpy as np
import pandas as pd

from elasticdl.python.tests.test_module import (
    feature_columns_fn,
    custom_sequential_model,
    custom_model_with_embedding,
)
from elasticdl.python.common.model_handler import ModelHander
from elasticdl.python.elasticdl.layers.embedding import Embedding

_model_zoo_path = os.path.dirname(os.path.realpath(__file__))


def _get_dataset():
    y_labels = np.array([1, 1, 0, 0, 1])
    x_data = pd.DataFrame({'age': [14, 56, 78, 38, 80],
                           'education': ['Bachelors', 'Master', 'Some-college',
                                         'Bachelors', 'Master']})
    dataset = tf.data.Dataset.from_tensor_slices((dict(x_data), y_labels))
    dataset = dataset.shuffle(len(x_data)).batch(4)
    return dataset


class DefaultModelHanderTest(unittest.TestCase):
    def setUp(self):
        self.model_handler = ModelHander.get_model_handler()

    def test_generate_train_model_for_elasticdl(self):
        model_inst = custom_model_with_embedding()
        model_inst = self.model_handler.generate_train_model_for_elasticdl(
            model_inst
        )
        self.assertEqual(type(model_inst.layers[1]),
                         tf.keras.layers.Embedding)

    def test_get_saved_model_from_trained_model(self):
        feature_columns = feature_columns_fn()
        model_inst = custom_sequential_model(feature_columns)
        dataset = _get_dataset()
        self.model_handler.get_saved_model_from_trained_model(
            model_inst, dataset)
        self.assertEqual(list(model_inst.inputs.keys()), [
                         'age', 'education'])
        self.assertEqual(
            model_inst.outputs[0].name, "sequential/Identity:0")


class ParameterSeverModelHandlerTest(unittest.TestCase):
    def setUp(self):
        self.model_handler = ModelHander.get_model_handler(
            distribution_strategy="ParameterServerStrategy"
        )

    def test_generate_train_model_for_elasticdl(self):
        model_inst = custom_model_with_embedding()
        model_inst = self.model_handler.generate_train_model_for_elasticdl(
            model_inst
        )
        self.assertEqual(type(model_inst.layers[1]), Embedding)

    def test_get_saved_model_from_trained_model(self):
        pass


if __name__ == "__main__":
    unittest.main()
