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

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.args import parse_master_args
from elasticdl.python.master.elasticdl_job_service import ElasticdlJobService
from elasticdl.python.tests.test_utils import (
    DatasetName,
    TaskManager,
    create_recordio_file,
)
from elasticdl_client.common.constants import DistributionStrategy


class ElasticdlJobServiceTest(unittest.TestCase):
    def setUp(self):
        self._model_zoo_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../../../model_zoo"
        )
        self.arguments = {
            "num_ps_pods": "1",
            "num_workers": "2",
            "job_type": str(elasticai_api_pb2.TRAINING),
            "minibatch_size": "32",
            "model_zoo": self._model_zoo_path,
            "model_def": "mnist.mnist_functional_api.custom_model",
            "job_name": "test",
            "worker_image": "ubuntu:18.04",
        }
        self._num_records = 128

    def _get_args(self):
        args = []
        for key, value in self.arguments.items():
            args.append("--" + key)
            args.append(value)
        return args

    def test_create_master_for_allreduce(self):
        self.arguments[
            "distribution_strategy"
        ] = DistributionStrategy.ALLREDUCE
        with tempfile.TemporaryDirectory() as temp_dir_name:
            create_recordio_file(
                self._num_records,
                DatasetName.TEST_MODULE,
                1,
                temp_dir=temp_dir_name,
            )
            self.arguments["training_data"] = temp_dir_name
            self.arguments["custom_training_loop"] = "true"
            args = self._get_args()
            args = parse_master_args(args)
            master = ElasticdlJobService(args, TaskManager(args))
            self.assertIsNotNone(master)

    def test_create_master_without_eval(self):
        self.arguments[
            "distribution_strategy"
        ] = DistributionStrategy.ALLREDUCE
        self.arguments["custom_training_loop"] = "true"
        self.arguments["model_def"] = "mnist.mnist_train_tfv2.train"
        with tempfile.TemporaryDirectory() as temp_dir_name:
            create_recordio_file(
                self._num_records,
                DatasetName.TEST_MODULE,
                1,
                temp_dir=temp_dir_name,
            )
            self.arguments["training_data"] = temp_dir_name
            args = self._get_args()
            args = parse_master_args(args)
            master = ElasticdlJobService(args, TaskManager(args))
            self.assertIsNone(master.evaluation_service)


if __name__ == "__main__":
    unittest.main()
