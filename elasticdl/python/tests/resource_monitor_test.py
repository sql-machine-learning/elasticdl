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

from elasticdl.python.master.resource_monitor import WorkerResourceMonitor


class ResourceMonitorTest(unittest.TestCase):
    def testWorkerResourceMonitor(self):
        resource_monitor = WorkerResourceMonitor()
        resource_monitor.set_worker_resource(2 * 1024 * 1024, 0.5)
        self.assertEqual(resource_monitor.get_worker_memory(), 2)


if __name__ == "__main__":
    unittest.main()
