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

"""In process master for unittests"""


class InProcessMaster(object):
    def __init__(self, master):
        self._m = master

    def get_task(self, req):
        return self._m.get_task(req, None)

    """
    def ReportGradient(self, req):
        for callback in self._callbacks:
            if test_call_back.ON_REPORT_GRADIENT_BEGIN in callback.call_times:
                callback()
        return self._m.ReportGradient(req, None)
    """

    def report_evaluation_metrics(self, req):
        return self._m.report_evaluation_metrics(req, None)

    def report_task_result(self, req):
        return self._m.report_task_result(req, None)

    def report_version(self, req):
        return self._m.report_version(req, None)
