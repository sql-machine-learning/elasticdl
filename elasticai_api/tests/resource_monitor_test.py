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
import json
import os
import time
import unittest

from elasticai_api.common.resource_monitor import ResourceMonitor


class ResourceMonitorTest(unittest.TestCase):
    def test_resource_monitor(self):
        TF_CONFIG = {
            "cluster": {
                "chief": ["localhost:2221"],
                "worker": {},
                "ps": ["localhost:2226"],
            },
            "task": {"type": "chief", "index": 0},
        }
        os.environ["TF_CONFIG"] = json.dumps(TF_CONFIG)
        resource_monitor = ResourceMonitor()
        time.sleep(0.3)
        self.assertTrue(resource_monitor._max_memory > 0.0)
        self.assertTrue(resource_monitor._max_cpu_percent >= 0.0)


if __name__ == "__main__":
    unittest.main()
