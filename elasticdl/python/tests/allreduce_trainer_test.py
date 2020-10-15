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

import horovod.tensorflow as hvd
import tensorflow as tf

from elasticdl.python.tests.test_module import custom_model, loss, optimizer
from elasticdl.python.worker.allreduce_trainer import (
    AllReduceTrainer,
    RendevousManager,
)


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
        self._trainer = AllReduceTrainer(master_client, "", model)

    def test_training_process(self):
        self._trainer.init_horovod_if_needed()
        features = tf.constant([[0.5], [0.6], [0.7]])
        labels = tf.constant([[1.0], [0.0], [1.0]])
        loss = self._trainer._training_process(features, labels)
        self.assertIsNotNone(loss)

    def test_train_minibatch(self):
        self._trainer.init_horovod_if_needed()
        features = tf.constant([[0.5], [0.6], [0.7]])
        labels = tf.constant([[1.0], [0.0], [1.0]])
        _, version, loss = self._trainer.train_minibatch(features, labels)
        self.assertEqual(version, 1)
        self.assertIsNotNone(loss)
        hvd.shutdown()

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
        self._manager = RendevousManager(master_client, "")

    def test_init_variables_if_needed(self):
        self._manager.init_horovod_if_needed()
        self.assertEqual(self._manager._rendezvous_id, 1)
        self.assertTrue(self.need_broadcast)


if __name__ == "__main__":
    unittest.main()
