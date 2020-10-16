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
from elasticdl.python.master.master import Master
from elasticdl.python.tests.test_utils import DatasetName, create_recordio_file
from elasticdl_client.common.constants import DistributionStrategy


class MasterTest(unittest.TestCase):
    def setUp(self):
        self._model_zoo_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../../../model_zoo"
        )
        self.arguments = [
            "--num_ps_pods",
            "1",
            "--num_workers",
            "2",
            "--job_type",
            elasticdl_pb2.TRAINING,
            "--minibatch_size",
            "32",
            "--model_zoo",
            self._model_zoo_path,
            "--model_def",
            "mnist.mnist_functional_api.custom_model",
            "--job_name",
            "test",
            "--worker_image",
            "ubuntu:18.04",
        ]

    def test_create_instance_manager(self):
        arguments = [
            "--distribution_strategy",
            DistributionStrategy.PARAMETER_SERVER,
        ]
        arguments.extend(self.arguments)
        num_records = 128
        with tempfile.TemporaryDirectory() as temp_dir_name:
            create_recordio_file(
                num_records, DatasetName.TEST_MODULE, 1, temp_dir=temp_dir_name
            )
            arguments.extend(["--training_data", temp_dir_name])
            args = parse_master_args(arguments)
            master = Master(args)
            self.assertIsNotNone(master.instance_manager)

    def test_create_master_for_allreduce(self):
        arguments = [
            "--distribution_strategy",
            DistributionStrategy.ALLREDUCE,
        ]
        arguments.extend(self.arguments)
        num_records = 128
        with tempfile.TemporaryDirectory() as temp_dir_name:
            create_recordio_file(
                num_records, DatasetName.TEST_MODULE, 1, temp_dir=temp_dir_name
            )
            arguments.extend(["--training_data", temp_dir_name])
            arguments.extend(["--custom_training_loop", "true"])
            args = parse_master_args(arguments)
            master = Master(args)
            self.assertIsNotNone(master.instance_manager)
            self.assertIsNone(master.callbacks_list)


if __name__ == "__main__":
    unittest.main()
