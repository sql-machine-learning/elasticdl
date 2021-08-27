# Copyright 2021 The ElasticDL Authors. All rights reserved.
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

import threading


class WorkerResourceMonitor(object):
    """Monitor the used resource of pods reported by workers
    """

    def __init__(self, chief_worker_name=None):
        self._worker_memory = 0.0
        self._worker_num = 0
        self._worker_cpu_percent = 0
        self._ps_memory = 0.0
        self._ps_num = 0
        self._ps_cpu_percent = 0
        self._chief_worker_name = chief_worker_name
        self._lock = threading.Lock()

    def get_worker_memory(self):
        return self._worker_memory

    def set_worker_resource(self, memory, cpu_percent):
        """Set memory and cpu_percent.
        Args:
            memory: Unit Bytes.
            cpu_percent: A float value in [0, 1)
        """
        memory = int(memory / 1024 / 1024)
        with self._lock:
            if memory > self._worker_memory:
                self._worker_memory = memory
            if cpu_percent > self._worker_cpu_percent:
                self._worker_cpu_percent = cpu_percent

    def set_chief_worker_name(self, pod_name):
        self._chief_worker_name = pod_name
