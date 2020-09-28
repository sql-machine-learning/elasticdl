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
        self._trainer = AllReduceTrainer(master_client, "", model, loss)

    def test_training_process_with_fault_tolerance(self):
        self._trainer.init_horovod_if_needed()
        features = tf.constant([[0.5], [0.6], [0.7]])
        labels = tf.constant([[1.0], [0.0], [1.0]])
        version, _ = self._trainer.training_process_with_fault_tolerance(
            features, labels
        )
        # Firstly, we will call model locally to create variables
        # for the model and optimizer. Then, we will train the
        # model using Horovod. So, the iteration step = 2.
        self.assertEqual(version, 2)
        hvd.shutdown()


if __name__ == "__main__":
    unittest.main()
