import unittest

import numpy as np
import tensorflow as tf

from elasticdl.python.common.constants import DistributionStrategy
from elasticdl.python.common.model_handler import ModelHandler
from elasticdl.python.elasticdl.layers.embedding import Embedding
from elasticdl.python.master.checkpoint_service import CheckpointService
from elasticdl.python.master.servicer import MasterServicer


class CustomModel(tf.keras.models.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(4, 2)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        embedding = self.embedding(inputs)
        output = self.dense(embedding)
        return output


def custom_model_with_embedding():
    inputs = tf.keras.layers.Input(shape=(4,), name="x")
    embedding = tf.keras.layers.Embedding(4, 2)(inputs)
    outputs = tf.keras.layers.Dense(1)(embedding)
    return tf.keras.models.Model(inputs, outputs)


def custom_sequential_model(feature_columns):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.DenseFeatures(feature_columns=feature_columns),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def feature_columns_fn():
    age = tf.feature_column.numeric_column("age", dtype=tf.int64)
    education = tf.feature_column.categorical_column_with_hash_bucket(
        "education", hash_bucket_size=4
    )
    education_one_hot = tf.feature_column.indicator_column(education)
    return [age, education_one_hot]


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


def _mock_model_trained_params(model):
    trained_params = {}
    for var in model.trainable_variables:
        trained_params[var.name] = np.ones(
            var.shape.as_list(), dtype="float32"
        )
    return trained_params


class DefaultModelHandlerTest(unittest.TestCase):
    def setUp(self):
        self.model_handler = ModelHandler.get_model_handler()

    def test_get_model_to_ps(self):
        model_inst = custom_model_with_embedding()
        model_inst = self.model_handler.get_model_to_train(model_inst)
        self.assertEqual(type(model_inst.layers[1]), tf.keras.layers.Embedding)

    def test_get_model_to_export(self):
        dataset = _get_dataset()
        feature_columns = feature_columns_fn()
        model_inst = custom_sequential_model(feature_columns)
        model_inst._build_model_with_inputs(inputs=dataset, targets=None)
        model_inst = self.model_handler.get_model_to_export(
            model_inst, dataset
        )
        self.assertEqual(list(model_inst.inputs.keys()), ["age", "education"])
        self.assertEqual(len(model_inst.outputs), 1)

        mock_params = _mock_model_trained_params(model_inst)
        for var in model_inst.trainable_variables:
            var.assign(mock_params[var.name])

        test_data = {
            "age": [14, 56, 78, 38, 80],
            "education": [
                "Bachelors",
                "Master",
                "Some-college",
                "Bachelors",
                "Master",
            ],
        }
        result = model_inst.call(test_data).numpy()
        self.assertEqual(result.tolist(), np.ones((5, 1)).tolist())


class ParameterSeverModelHandlerTest(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        self.model_handler = ModelHandler.get_model_handler(
            distribution_strategy=DistributionStrategy.PARAMETER_SERVER,
            checkpoint_dir="elasticdl/python/tests/testdata/functional_ckpt/",
        )

    def test_get_model_to_train(self):
        model_inst = custom_model_with_embedding()
        model_inst = self.model_handler.get_model_to_train(model_inst)
        self.assertEqual(type(model_inst.layers[1]), Embedding)

    def test_get_model_to_export(self):
        model_inst = custom_model_with_embedding()
        train_model = self.model_handler.get_model_to_train(model_inst)
        export_model = self.model_handler.get_model_to_export(
            train_model, dataset=None
        )

        test_data = tf.constant([0])
        result = export_model.call(test_data).numpy()
        self.assertEqual(result[0][0], 3.0)

    def test_get_subclass_model_to_export(self):
        self.model_handler._checkpoint_dir = (
            "elasticdl/python/tests/testdata/subclass_ckpt/"
        )

        def _get_dataset():
            dataset = tf.data.Dataset.from_tensor_slices(
                np.random.randint(0, 10, (10, 4))
            )
            dataset = dataset.batch(2)
            return dataset

        model_inst = CustomModel()
        dataset = _get_dataset()

        train_model = self.model_handler.get_model_to_train(model_inst)
        self.assertEqual(type(train_model.embedding), Embedding)

        export_model = self.model_handler.get_model_to_export(
            train_model, dataset=dataset
        )

        test_data = tf.constant([0])
        result = export_model.call(test_data).numpy()
        self.assertEqual(result[0][0], 3.0)


if __name__ == "__main__":
    unittest.main()
