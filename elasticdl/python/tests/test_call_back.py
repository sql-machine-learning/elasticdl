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

"""Callback for unittests."""
from abc import ABC, abstractmethod

ON_REPORT_GRADIENT_BEGIN = "on_report_gradient_begin"
ON_REPORT_EVALUATION_METRICS_BEGIN = "on_report_evaluation_metrics_begin"


class BaseCallback(ABC):
    """Base class of callbacks used for testing."""

    def __init__(self, master, worker, call_times):
        self._master = master
        self._worker = worker
        self.call_times = call_times

    @abstractmethod
    def __call__(self):
        pass
