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

TIMING_GROUP = [
    "task_process",
    "batch_process",
    "get_model",
    "report_gradient",
]


class Timing(object):
    def __init__(self, enable, logger=None):
        self.enable = enable
        self.logger = logger
        self.timings = {}
        self.start_time = {}
        self.reset()

    def reset(self):
        if not self.enable:
            return
        for timing_type in TIMING_GROUP:
            self.timings[timing_type] = 0

    def start_record_time(self, timing_type):
        if not self.enable:
            return
        self.start_time[timing_type] = time.time()

    def end_record_time(self, timing_type):
        if not self.enable:
            return
        self.timings[timing_type] += time.time() - self.start_time[timing_type]

    def report_timing(self, reset=False):
        if not self.enable:
            return
        for timing_type in TIMING_GROUP:
            self.logger.debug(
                "%s time is %.6g seconds"
                % (timing_type, self.timings[timing_type])
            )
        if reset:
            self.reset()
