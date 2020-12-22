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

import horovod.torch as hvd
import torch.optim as optim

from elasticai_api.pytorch.optimizer import DistributedOptimizer
from elasticdl.python.tests.test_module import TorchModel


class ElasticOptimizerTest(unittest.TestCase):
    def test_step(self):
        hvd.init()
        model = TorchModel()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        optimizer = DistributedOptimizer(
            optimizer, fixed_global_batch_size=True
        )
        optimizer.zero_grad()
        optimizer.backward_passes_per_step = 2
        optimizer.step()
        self.assertEqual(optimizer._iter_step, 1)
        self.assertFalse(optimizer._update_gradients)

        optimizer.step()
        self.assertEqual(optimizer._iter_step, 2)
        self.assertTrue(optimizer._update_gradients)


if __name__ == "__main__":
    unittest.main()
