import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.constants import DistributionStrategy, JobType
from elasticdl.python.common.model_handler import ModelHandler
from elasticdl.python.elasticdl.callbacks.saved_model_exporter import (
    SavedModelExporter,
)
from elasticdl.python.tests.test_utils import save_checkpoint_without_embedding
from elasticdl.python.worker.task_data_service import TaskDataService


def custom_model_with_embedding_layer():
    inputs = tf.keras.layers.Input(shape=(4,), name="x")
    embedding = tf.keras.layers.Embedding(300000, 2)(inputs)
    outputs = tf.keras.layers.Dense(1)(embedding)
    return tf.keras.models.Model(inputs, outputs)


def dataset_fn(dataset, mode, metadata):
    return dataset


class MockWorker:
    def __init__(self):
        self._custom_data_reader = None


class SavedModelExporterTest(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()

    def test_on_train_end(self):
        worker = MockWorker()
        task_data_service = TaskDataService(
            worker, JobType.TRAINING_WITH_EVALUATION
        )
        dataset = tf.data.Dataset.from_tensor_slices(
            np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        )
        task_data_service._pending_train_end_callback_task = (
            "",
            0,
            1,
            elasticdl_pb2.TRAIN_END_CALLBACK,
        )
        task_data_service.get_dataset_by_task = mock.Mock(return_value=dataset)

        with tempfile.TemporaryDirectory() as temp_dir_name:
            checkpoint_dir = os.path.join(temp_dir_name, "checkpoint")
            model = custom_model_with_embedding_layer()
            save_checkpoint_without_embedding(model, checkpoint_dir)
            model_handler = ModelHandler.get_model_handler(
                distribution_strategy=DistributionStrategy.PARAMETER_SERVER,
                checkpoint_dir=checkpoint_dir,
            )
            saved_model_exporter = SavedModelExporter(
                task_data_service, dataset_fn, model_handler
            )
            saved_model_path = os.path.join(temp_dir_name, "test_exporter")
            params = {"batch_size": 10, "saved_model_path": saved_model_path}
            saved_model_exporter.set_params(params)
            saved_model_exporter.set_model(model)
            saved_model_exporter.on_train_end()
            self.assertTrue(os.path.exists(saved_model_path))
            self.assertTrue(
                os.path.exists(
                    os.path.join(saved_model_path, "saved_model.pb")
                )
            )


class MaxStepsStoppingTest(unittest.TestCase):
    def test_on_task_end(self):
        max_steps_stopping = MaxStepsStopping(max_steps=20)
        max_steps_stopping.set_model(tf.keras.Model())
        max_steps_stopping.model.stop_training = False
        max_steps_stopping.set_params({"batch_size": 128})
        for i in range(6):
            start = i * 512
            end = (i + 1) * 512
            task = _Task(
                shard_name="test",
                start=start,
                end=end,
                type=elasticdl_pb2.TRAINING,
            )
            max_steps_stopping.on_task_end(task)
        self.assertTrue(max_steps_stopping.model.stop_training)


if __name__ == "__main__":
    unittest.main()
