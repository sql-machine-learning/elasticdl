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

import time
import unittest


class LearningRateTest(unittest.TestCase):
    @staticmethod
    def get_lr(lr_modulation, opt, multiplier):
        lr_modulation.set_multiplier(multiplier)
        # sleep 1s to wait that all threads are in this method call
        time.sleep(1)
        return opt.learning_rate

    @staticmethod
    def apply_gradients_with_modulation(
        lr_modulation, opt, multiplier, variables, grads
    ):
        grads_and_vars = zip(grads, variables)
        lr_modulation.set_multiplier(multiplier)
        # sleep 1s to wait that all threads are in this method call
        time.sleep(1)
        opt.apply_gradients(grads_and_vars)
        return [v.numpy() for v in variables]


if __name__ == "__main__":
    unittest.main()
