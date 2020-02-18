import os
import tempfile
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_v2 as fc_lib

from elasticdl.python.common.constants import DistributionStrategy
from elasticdl.python.common.model_handler import ModelHandler
from elasticdl.python.common.save_utils import CheckpointSaver
from elasticdl.python.elasticdl.feature_column.feature_column import (
    EmbeddingColumn,
)
from elasticdl.python.elasticdl.layers.embedding import Embedding
from elasticdl.python.keras.layers import SparseEmbedding
from elasticdl.python.ps.parameters import Parameters

EMBEDDING_INPUT_DIM = 300000


class CustomModel(tf.keras.models.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(EMBEDDING_INPUT_DIM, 2)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        embedding = self.embedding(inputs)
        output = self.dense(embedding)
        return output


def custom_model_with_embedding_layer():
    inputs = tf.keras.layers.Input(shape=(4,), name="x")
    embedding = tf.keras.layers.Embedding(EMBEDDING_INPUT_DIM, 2)(inputs)
    outputs = tf.keras.layers.Dense(1)(embedding)
    return tf.keras.models.Model(inputs, outputs)


def custom_model_with_embedding_column():
    age = tf.feature_column.numeric_column("age", dtype=tf.int64)
    user_id_embedding = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket(
            "user_id", hash_bucket_size=EMBEDDING_INPUT_DIM
        ),
        dimension=8,
    )
    return custom_sequential_model([age, user_id_embedding])


def custom_model_with_sparse_embedding():
    sparse_input = tf.keras.layers.Input(
        shape=(4,), dtype="int64", sparse=True, name="sparse_feature"
    )
    embedding = SparseEmbedding(
        EMBEDDING_INPUT_DIM, 2, combiner="sum", name="embedding"
    )(sparse_input)
    outputs = tf.keras.layers.Dense(1)(embedding)
    return tf.keras.models.Model(sparse_input, outputs)


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
        model_inst = custom_model_with_embedding_layer()
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
            checkpoint_dir="",
        )

    def _save_model(self, is_subclass):
        prefix = "custom_model/" if is_subclass else ""
        ckpt_dir = self.model_handler._checkpoint_dir
        checkpoint_saver = CheckpointSaver(ckpt_dir, 0, 0, False)
        params = Parameters()
        params.non_embedding_params[
            prefix + "embedding/embeddings:0"
        ] = tf.Variable(tf.ones([EMBEDDING_INPUT_DIM, 2]), dtype=tf.float32)
        params.non_embedding_params[prefix + "dense/kernel:0"] = tf.Variable(
            [[1.0], [1.0]]
        )
        params.non_embedding_params[prefix + "dense/bias:0"] = tf.Variable(
            [1.0]
        )
        params.version = 100
        model_pb = params.to_model_pb()
        checkpoint_saver.save(100, model_pb, False)

    def test_get_model_with_embedding_layer_to_train(self):
        model_inst = custom_model_with_embedding_layer()
        model_inst = self.model_handler.get_model_to_train(model_inst)
        self.assertEqual(type(model_inst.layers[1]), Embedding)

    def test_get_model_with_embedding_column_to_train(self):
        model_inst = custom_model_with_embedding_column()
        self.assertEqual(
            type(model_inst.layers[0]._feature_columns[1]),
            fc_lib.EmbeddingColumn,
        )
        model_inst = self.model_handler.get_model_to_train(model_inst)
        self.assertEqual(
            type(model_inst.layers[0]._feature_columns[1]), EmbeddingColumn
        )

    def test_get_model_to_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.model_handler._checkpoint_dir = os.path.join(
                temp_dir, "test_export"
            )
            self._save_model(False)
            model_inst = custom_model_with_embedding_layer()
            train_model = self.model_handler.get_model_to_train(model_inst)
            export_model = self.model_handler.get_model_to_export(
                train_model, dataset=None
            )

            test_data = tf.constant([0])
            result = export_model.call(test_data).numpy()
            self.assertEqual(result[0][0], 3.0)

    def test_get_subclass_model_to_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.model_handler._checkpoint_dir = os.path.join(
                temp_dir, "test_export"
            )
            self._save_model(True)

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

    def test_get_model_with_sparse_to_train(self):
        model_inst = custom_model_with_sparse_embedding()
        model_inst = self.model_handler.get_model_to_train(model_inst)
        self.assertEqual(type(model_inst.layers[1]), Embedding)

    def test_get_model_with_sparse_to_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.model_handler._checkpoint_dir = os.path.join(
                temp_dir, "test_export"
            )
            self._save_model(False)
            model_inst = custom_model_with_sparse_embedding()
            train_model = self.model_handler.get_model_to_train(model_inst)

            # Model handler will restore model parameters from the checkpoint
            # directory and assign parameters to train_model.
            export_model = self.model_handler.get_model_to_export(
                train_model, dataset=None
            )
            test_data = tf.SparseTensor(
                indices=[[0, 0]], values=[0], dense_shape=(1, 1)
            )
            result = export_model.call(test_data).numpy()

            # The embedding table in checkpoint file is
            # [[1.0, 1.0], [1.0, 1.0], [1.0,1.0], [1.0, 1.0]], weights in the
            # dense layer is [[1.0],[1.0]], bias is [1.0]. So the result
            # is 3.0.
            self.assertEqual(result[0][0], 3.0)


if __name__ == "__main__":
    unittest.main()
