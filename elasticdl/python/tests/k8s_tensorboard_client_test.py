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
import random
import time
import unittest

from elasticdl.python.common.k8s_tensorboard_client import TensorBoardClient


@unittest.skipIf(
    os.environ.get("K8S_TESTS", "True") == "False",
    "No Kubernetes cluster available",
)
class K8sTensorBoardClientTest(unittest.TestCase):
    def test_create_tensorboard_service(self):
        tb_client = TensorBoardClient(
            image_name=None,
            namespace="default",
            job_name="test-job-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            event_callback=None,
        )
        tb_client._k8s_client.create_tensorboard_service(
            port=80, service_type="LoadBalancer"
        )
        time.sleep(1)
        service = tb_client._get_tensorboard_service()
        self.assertTrue("load_balancer" in service["status"])
        self.assertEqual(service["spec"]["ports"][0]["port"], 80)


if __name__ == "__main__":
    unittest.main()
