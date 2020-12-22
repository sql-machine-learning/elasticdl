# Copyright 2020 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

from elasticai_api.proto import elasticai_api_pb2
from elasticdl.python.common.model_handler import ModelHandler
from elasticdl.python.common.save_utils import (
    save_checkpoint_without_embedding,
)
from elasticdl.python.elasticdl.callbacks import (
    LearningRateScheduler,
    SavedModelExporter,
)
from elasticdl.python.worker.task_data_service import TaskDataService
from elasticdl_client.common.constants import DistributionStrategy


def custom_model_with_embedding_layer():
    inputs = tf.keras.layers.Input(shape=(4,), name="x")
    embedding = tf.keras.layers.Embedding(300000, 2)(inputs)
    outputs = tf.keras.layers.Dense(1)(embedding)
    return tf.keras.models.Model(inputs, outputs)


def feed(dataset, mode, metadata):
    return dataset


class MockWorker:
    def __init__(self):
        self._custom_data_reader = None


class SavedModelExporterTest(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()

    def test_on_train_end(self):
        worker = MockWorker()
        task_data_service = TaskDataService(worker)
        dataset = tf.data.Dataset.from_tensor_slices(
            np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        )
        task_data_service.train_end_callback_task = (
            "",
            0,
            1,
            elasticai_api_pb2.TRAIN_END_CALLBACK,
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
                task_data_service, feed, model_handler
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


class LearningRateSchedulerTest(unittest.TestCase):
    def _schedule(self, model_version):
        return 0.2 if model_version < 2 else 0.1

    def test_raise_error(self):
        def _schedule(model_version):
            return 1 if model_version < 2 else 2

        learning_rate_scheduler = LearningRateScheduler(_schedule)
        model = tf.keras.Model()
        learning_rate_scheduler.set_model(model)
        with self.assertRaises(ValueError):
            learning_rate_scheduler.on_train_batch_begin(batch=1)

        model.optimizer = tf.optimizers.SGD(0.1)
        with self.assertRaises(ValueError):
            learning_rate_scheduler.on_train_batch_begin(batch=1)

    def test_learning_rate_scheduler(self):
        learning_rate_scheduler = LearningRateScheduler(self._schedule)
        model = tf.keras.Model()
        model.optimizer = tf.optimizers.SGD(0.1)
        learning_rate_scheduler.set_model(model)

        learning_rate_scheduler.on_train_batch_begin(batch=1)
        self.assertEqual(model.optimizer.lr.numpy(), np.float32(0.2))
        learning_rate_scheduler.on_train_batch_begin(batch=2)
        self.assertEqual(model.optimizer.lr.numpy(), np.float32(0.1))

        model_versions = [0, 1, 2]
        variables = []
        grads = []
        original_values = [1.2, 0.8]
        grad_values = [0.2, 0.1]

        for i in range(len(model_versions)):
            variables.append([tf.Variable(v) for v in original_values])
            grads.append([tf.convert_to_tensor(g) for g in grad_values])

        results = []
        for i in range(len(model_versions)):
            result = self.apply_gradients_with_scheduler(
                learning_rate_scheduler,
                model.optimizer,
                model_versions[i],
                variables[i],
                grads[i],
            )
            results.append(result)

        place = 5
        for i in range(0, len(model_versions)):
            i_diff = [
                original_values[j] - results[i][j]
                for j in range(len(original_values))
            ]
            for j in range(len(original_values)):
                # variable value change ratio equals the learning rate ratio
                # for SGD without momentum
                self.assertAlmostEqual(
                    i_diff[j],
                    grad_values[j] * self._schedule(model_versions[i]),
                    place,
                )

    @staticmethod
    def apply_gradients_with_scheduler(
        lr_scheduler, opt, model_version, variables, grads
    ):
        lr_scheduler.on_train_batch_begin(model_version)
        grads_and_vars = zip(grads, variables)
        opt.apply_gradients(grads_and_vars)
        return [v.numpy() for v in variables]


if __name__ == "__main__":
    unittest.main()
