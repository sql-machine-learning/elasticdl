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
import unittest

import tensorflow as tf

from elasticdl.python.common.args import parse_worker_args
from elasticdl.python.worker.worker import Worker


class WorkerTest(unittest.TestCase):
    def setUp(self):
        self._model_zoo_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../../../model_zoo"
        )
        self._batch_size = 16

    def _create_worker(self, arguments):
        tf.keras.backend.clear_session()
        tf.random.set_seed(22)
        args = parse_worker_args(arguments)
        return Worker(args)


if __name__ == "__main__":
    unittest.main()
