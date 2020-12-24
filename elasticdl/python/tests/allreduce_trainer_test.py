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

import unittest
from unittest.mock import MagicMock, Mock

import tensorflow as tf
import torch.optim as optim

from elasticai_api.common.base_controller import (
    AllReduceController,
    RendevousManager,
)
from elasticai_api.common.data_shard_service import DataShardService
from elasticai_api.pytorch.controller import (
    PyTorchAllReduceController,
    create_elastic_controller,
)
from elasticai_api.pytorch.optimizer import DistributedOptimizer
from elasticai_api.tensorflow.controller import TensorFlowV2AllReduceController
from elasticdl.python.tests.test_module import (
    TorchModel,
    custom_model,
    loss,
    optimizer,
)
from elasticdl.python.worker.allreduce_trainer import AllReduceTrainer


class AllReduceTrainerTest(unittest.TestCase):
    def setUp(self):
        master_client = Mock()
        master_client.get_comm_rank = MagicMock(
            return_value=Mock(
                rendezvous_id=1, rank_id=0, world_size=1, rendezvous_port=0
            )
        )
        model = custom_model()
        model.optimizer = optimizer()
        model.loss = loss
        self._trainer = AllReduceTrainer(master_client, model)

    def test_training_process(self):
        self._trainer._rendezvous_manager.init_horovod_if_needed()
        features = tf.constant([[0.5], [0.6], [0.7]])
        labels = tf.constant([[1.0], [0.0], [1.0]])
        loss = self._trainer._training_process(features, labels)
        self.assertIsNotNone(loss)

    def test_train_minibatch(self):
        self._trainer._rendezvous_manager.init_horovod_if_needed()
        features = tf.constant([[0.5], [0.6], [0.7]])
        labels = tf.constant([[1.0], [0.0], [1.0]])
        _, version, loss = self._trainer.train_minibatch(features, labels)
        self.assertEqual(version, 1)
        self.assertIsNotNone(loss)

    def test_init_variables_if_needed(self):
        features = tf.constant([[0.5], [0.6], [0.7]])
        labels = tf.constant([[1.0], [0.0], [1.0]])
        self._trainer.init_variables_if_need(features, labels)
        self.assertTrue(self._trainer._var_created)
        self.assertEqual(self._trainer._optimizer.iterations.numpy(), 0)


class RendevousManagerTest(unittest.TestCase):
    def setUp(self):
        master_client = Mock()
        master_client.get_comm_rank = MagicMock(
            return_value=Mock(
                rendezvous_id=1, rank_id=0, world_size=1, rendezvous_port=0
            )
        )
        self._manager = RendevousManager(master_client)

    def test_init_variables_if_needed(self):
        self._manager.init_horovod_if_needed()
        self.assertEqual(self._manager._rendezvous_id, 1)
        self.assertTrue(self._manager.need_broadcast)


class AllReduceControllerTest(unittest.TestCase):
    def train(self):
        pass

    def test_elastic_run(self):
        master_client = Mock()
        master_client.get_comm_rank = MagicMock(
            return_value=Mock(
                rendezvous_id=1, rank_id=0, world_size=1, rendezvous_port=0
            )
        )
        data_shard_service = DataShardService(master_client, 1)
        controller = AllReduceController(master_client, data_shard_service)
        elastic_run = controller.elastic_run(self.train)
        elastic_run()
        self.assertFalse(controller._first_call)


class TensorFlowV2ReduceControllerTest(unittest.TestCase):
    def setUp(self):
        master_client = Mock()
        master_client.get_comm_rank = MagicMock(
            return_value=Mock(
                rendezvous_id=1, rank_id=0, world_size=1, rendezvous_port=0
            )
        )
        data_shard_service = DataShardService(master_client, 1)
        self.controller = TensorFlowV2AllReduceController(
            master_client, data_shard_service
        )

    def _train(self):
        return 1

    def test_broadcast(self):
        self.controller.set_broadcast_model(tf.keras.Model())
        self.controller.set_broadcast_optimizer(tf.optimizers.SGD(0.01))
        self.controller.broadcast()
        self.assertIsNotNone(self.controller._model)
        self.assertIsNotNone(self.controller._optimizer)

    def test_train_one_batch_with_retries(self):
        self.controller.set_broadcast_model(tf.keras.Model())
        self.controller.set_broadcast_optimizer(tf.optimizers.SGD(0.01))
        result = self.controller.train_one_batch_with_retries(self._train)
        self.assertEqual(result, 1)


class PyTorchReduceControllerTest(unittest.TestCase):
    def setUp(self):
        master_client = Mock()
        master_client.get_comm_rank = MagicMock(
            return_value=Mock(
                rendezvous_id=1, rank_id=0, world_size=1, rendezvous_port=0
            )
        )
        data_shard_service = DataShardService(master_client, 1)
        self.controller = PyTorchAllReduceController(
            master_client, data_shard_service
        )

    def train(self):
        return 1

    def test_elastic_run(self):
        import horovod.torch as hvd

        model = TorchModel()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        hvd.init()
        optimizer = hvd.DistributedOptimizer(optimizer)
        self.controller.set_broadcast_model(model)
        self.controller.set_broadcast_optimizer(optimizer)
        self.controller.broadcast()
        result = self.controller.train_one_batch_with_retries(self.train)
        self.controller.restore()
        self.assertEqual(result, 1)
        self.assertIsNotNone(self.controller._model)
        self.assertIsNotNone(self.controller._optimizer)
        self.assertEqual(self.controller.global_completed_batch_num, 1)

    def test_create_elastic_controller(self):
        controller = create_elastic_controller(batch_size=64)
        self.assertIsNotNone(controller)
        self.assertIsNotNone(controller.data_shard_service._mc)
        self.assertEqual(controller.data_shard_service._batch_size, 64)

    def test_reset_backward_passes_per_step(self):
        model = TorchModel()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        optimizer = DistributedOptimizer(
            optimizer, fixed_global_batch_size=True
        )
        self.controller.set_broadcast_optimizer(optimizer)
        self.controller.global_batch_num_per_step = 2
        self.controller.reset_backward_passes_per_step()
        self.assertEqual(self.controller.backward_passes_per_step, 2)


if __name__ == "__main__":
    unittest.main()
