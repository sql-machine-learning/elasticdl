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

import collections

PodInfo = collections.namedtuple("PodInfo", ("type", "id", "name", "address"))

ClusterContext = collections.namedtuple("ClusterContext", ("pod_manager"))


class PodEventObserver(object):
    """
    The interface for the observers that are interested in the pod event.
    The subclass observers can override the following methods to handle
    various events.
    """

    def on_pod_started(self, pod_info, cluster_context):
        pass

    def on_pod_completed(self, pod_info, cluster_context):
        pass

    def on_pod_failed(self, pod_info, cluster_context):
        pass

    def on_pod_deleted(self, pod_info, cluster_context):
        pass
