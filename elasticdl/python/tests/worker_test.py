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

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.args import parse_worker_args
from elasticdl.python.worker.worker import Worker
from elasticdl_client.common.constants import DistributionStrategy


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

    def test_init_training_func_from_args(self):
        arguments = [
            "--worker_id",
            "0",
            "--job_type",
            elasticai_api_pb2.TRAINING,
            "--minibatch_size",
            self._batch_size,
            "--model_zoo",
            self._model_zoo_path,
            "--model_def",
            "mnist.mnist_train_tfv2.train",
            "--distribution_strategy",
            DistributionStrategy.ALLREDUCE,
            "--custom_training_loop",
            "true",
        ]
        worker = self._create_worker(arguments)
        self.assertIsNotNone(worker._feed)
        self.assertIsNotNone(worker._training_func)
        self.assertEqual(worker._minibatch_size, 16)
        self.assertIsNotNone(worker._task_data_service)


if __name__ == "__main__":
    unittest.main()
