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

import abc
import collections

PodInfo = collections.namedtuple(
    "PodInfo", ("type", "id", "name", "ip", "status", "start_time")
)

ClusterContext = collections.namedtuple("ClusterContext", ("pod_manager"))


class PodEventCallback(metaclass=abc.ABCMeta):
    """
    The interface for the observers that are interested in the pod event.
    The subclass observers can override the following methods to handle
    various events.
    """

    @abc.abstractmethod
    def on_pod_started(self, pod_info, cluster_context):
        """
        The handler for the pod started event.
        Args:
            pod_info: A PodInfo object. It's the pod that just becomes running.
            cluster_context: A ClusterContext object. It contains all the
                context information about the cluster for the job.
        """
        pass

    @abc.abstractmethod
    def on_pod_succeeded(self, pod_info, cluster_context):
        """
        The handler for the pod succeeded event.
        Args:
            pod_info: A PodInfo object. It's the pod that just terminates
                in success.
            cluster_context: A ClusterContext object. It contains all the
                context information about the cluster for the job.
        """
        pass

    @abc.abstractmethod
    def on_pod_failed(self, pod_info, cluster_context):
        """
        The handler for the pod failed event.
        Args:
            pod_info: A PodInfo object. It's the pod that just terminates
                in failure.
            cluster_context: A ClusterContext object. It contains all the
                context information about the cluster for the job.
        """
        pass

    @abc.abstractmethod
    def on_pod_deleted(self, pod_info, cluster_context):
        """
        The handler for the pod deleted event.
        Args:
            pod_info: A PodInfo object. It's the pod which is just deleted.
            cluster_context: A ClusterContext object. It contains all the
                context information about the cluster for the job.
        """
        pass


class TaskRescheduleCallback(PodEventCallback):
    def __init__(self, task_manager):
        super(TaskRescheduleCallback, self).__init__()
        self._task_manager = task_manager

    def on_pod_started(self, pod_info, cluster_context):
        pass

    def on_pod_succeeded(self, pod_info, cluster_context):
        pass

    def on_pod_failed(self, pod_info, cluster_context):
        if pod_info.id is not None:
            self._task_manager.recover_tasks(pod_info.id)

    def on_pod_deleted(self, pod_info, cluster_context):
        if pod_info.id is not None:
            self._task_manager.recover_tasks(pod_info.id)


class RendezvousServiceRefreshCallback(PodEventCallback):
    def __init__(self, rendezvous_server):
        super(RendezvousServiceRefreshCallback, self).__init__()
        self._rendezvous_server = rendezvous_server

    def on_pod_started(self, pod_info, cluster_context):
        pass

    def on_pod_succeeded(self, pod_info, cluster_context):
        self._rendezvous_server.remove_worker(pod_info.ip)

    def on_pod_failed(self, pod_info, cluster_context):
        self._rendezvous_server.remove_worker(pod_info.ip)

    def on_pod_deleted(self, pod_info, cluster_context):
        self._rendezvous_server.remove_worker(pod_info.ip)
