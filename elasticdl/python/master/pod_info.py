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


import copy
import math

from elasticdl.python.common.constants import PodStatus
from elasticdl.python.common.k8s_client import PodType


def _is_float_str(str_number):
    if not str_number:
        return False
    try:
        float(str_number)
        return True
    except ValueError:
        return False


def resource_str_to_dict(resource_str):
    """Convert the resource configuration like "memory=100Mi,cpu=5"
    to a dictionary {"memory": "100Mi", "cpu":"5"}."""
    resource_dict = {}
    for value in resource_str.strip().split(","):
        resource_dict[value.split("=")[0]] = value.split("=")[1]
    return resource_dict


def resource_dict_to_str(resource_dict):
    """Convert a dictionary lik dictionary {"memory": "100Mi", "cpu":"5"}
    to a string "memory=100Mi,cpu=5"."""
    values = []
    for name, value in resource_dict.items():
        values.append("{}={}".format(name, value))
    return ",".join(values)


class ResourceConfig(object):
    def __init__(
        self, resource_requests, resource_limits, priority, image_name
    ):
        self.resource_requests = resource_requests
        self.resource_limits = resource_limits
        self.priority = priority
        self.image_name = image_name

    def update_memory(self, memory_mi):
        """Update the memory config.
        Args:
            memory: The unit is Mi
        """
        memory_mi = str(int(memory_mi)) + "Mi"
        if self.resource_requests:
            requests = resource_str_to_dict(self.resource_requests)
            requests["memory"] = memory_mi
            self.resource_requests = resource_dict_to_str(requests)
        if self.resource_limits:
            limits = resource_str_to_dict(self.resource_limits)
            limits["memory"] = memory_mi
            self.resource_limits = resource_dict_to_str(limits)

    def get_memory_mi(self):
        resource = resource_str_to_dict(self.resource_requests)
        return int(resource["memory"][0:-2])


class TypedPodConfig(object):
    def __init__(self):
        self._default_config = ResourceConfig(
            resource_requests="cpu=1,memory=2048Mi",
            resource_limits="cpu=1,memory=2048Mi",
            priority=None,
            image_name=None,
        )
        self._configs = {}
        self._typed_pod_num = {}

    def add_typed_pod_config(
        self,
        pod_type,
        num,
        resource_requests,
        resource_limits,
        priority,
        image_name,
    ):
        self._typed_pod_num[pod_type] = num
        self._configs[pod_type] = ResourceConfig(
            resource_requests, resource_limits, priority, image_name
        )

    def get_typed_resource_config(self, pod_type):
        return self._configs.get(pod_type, self._default_config)

    def get_typed_pod_num(self, pod_type):
        return self._typed_pod_num.get(pod_type, 0)

    def get_pod_types(self):
        return list(self._configs.keys())


class PodInfo(object):
    def __init__(
        self,
        pod_type,
        id,
        name=None,
        pod_ip=None,
        node_ip=None,
        qos=None,
        status=PodStatus.INITIAL,
        start_time=None,
        original_index=None,
        relaunch_count=0,
        is_critical_pod=False,
        max_relaunch_count=0,
        relaunchable=True,
        resource_config=None,
    ):
        self.type = pod_type
        self.id = id
        self.name = name
        self.pod_ip = pod_ip
        self.node_ip = node_ip
        self.qos = qos
        self.status = status
        self.start_time = start_time
        self.original_index = (
            original_index if original_index is not None else id
        )
        self.relaunch_count = relaunch_count
        self.relaunchable = relaunchable
        self.is_critical_pod = is_critical_pod
        self.max_relaunch_count = max_relaunch_count
        self.resource_config = resource_config
        self.is_recovered_oom = False

    def inc_relaunch_count(self):
        self.relaunch_count += 1

    def update_info(
        self,
        name=None,
        pod_ip=None,
        node_ip=None,
        qos=None,
        status=None,
        start_time=None,
    ):
        if name is not None:
            self.name = name
        if pod_ip is not None:
            self.pod_ip = pod_ip
        if node_ip is not None:
            self.node_ip = node_ip
        if qos is not None:
            self.qos = qos
        if status is not None:
            self.status = status
        if start_time is not None:
            self.start_time = start_time

    def get_relaunch_pod_info(self, new_id):
        new_pod_info = copy.deepcopy(self)
        new_pod_info.id = new_id
        new_pod_info.name = None
        new_pod_info.status = PodStatus.INITIAL
        new_pod_info.start_time = None
        new_pod_info.pod_ip = None
        new_pod_info.node_ip = None
        return new_pod_info

    def is_unrecoverable_failure(self):
        if (
            self.is_critical_pod
            and self.relaunch_count >= self.max_relaunch_count
        ):
            return True
        return False

    def set_priority(self, priority):
        self.resource_config.priority = priority

    def set_memory(self, memory):
        self.resource_config.update_memory(memory)

    def increment_memory(self, config_memory):
        """Increment the memory to launch pod. The new memory
        is max(1.5 * memory, the memory set by users).

        Args:
            config_memory: int, with unit Mi
        """
        cur_memory = self.resource_config.get_memory_mi()
        new_memory = int(max(cur_memory * 1.5, config_memory))
        self.resource_config.update_memory(new_memory)


def init_pod_info(relaunch_on_worker_failure, typed_pod_config):
    """
    typed_pod_config: a dict with pod_type as key, the PodResourceConfig
                    as the value.
    relaunch_on_worker_failure: int, the number of relaunches.
    return: a dict with pod_type as key, and another dict as value.
            The other dict uses pod id as key, and PodInfo as value.
    """
    type_id_pod_info = {}
    for pod_type in typed_pod_config.get_pod_types():
        id_pod_info = {}
        resource_config = typed_pod_config.get_typed_resource_config(pod_type)
        typed_pod_num = typed_pod_config.get_typed_pod_num(pod_type)
        relaunchable = False if pod_type == PodType.PS else True
        for id in range(typed_pod_num):
            id_pod_info[id] = PodInfo(
                pod_type,
                id,
                max_relaunch_count=relaunch_on_worker_failure,
                relaunchable=relaunchable,
                resource_config=copy.deepcopy(resource_config),
            )
        type_id_pod_info[pod_type] = id_pod_info
    return type_id_pod_info


def set_critical_pod(pod_info, ps_is_critical=True, critical_worker_index={}):
    """
    pod_info is a dict, where pod_info[type][id] is a PodInfo instance
    Set is_critical_pod values accordingly
    """
    if PodType.PS in pod_info:
        for info in pod_info[PodType.PS].values():
            info.is_critical_pod = ps_is_critical
            info.max_relaunch_count = (
                0 if ps_is_critical else info.max_relaunch_count
            )
    if PodType.WORKER in pod_info:
        for id in pod_info[PodType.WORKER]:
            if id in critical_worker_index:
                pod_info[PodType.WORKER][id].is_critical_pod = True
                pod_info[PodType.WORKER][
                    id
                ].max_relaunch_count = critical_worker_index[id]
    if PodType.EVALUATOR in pod_info:
        for info in pod_info[PodType.EVALUATOR].values():
            info.is_critical_pod = True
    if PodType.CHIEF in pod_info:
        for info in pod_info[PodType.CHIEF].values():
            info.is_critical_pod = True


def set_worker_pod_priority(pod_info, worker_pod_priority):
    if PodType.WORKER not in pod_info:
        return
    num_workers = len(pod_info[PodType.WORKER])
    if _is_float_str(worker_pod_priority):
        fraction = float(worker_pod_priority)
        high_count = math.ceil(num_workers * fraction)
        for i in range(num_workers):
            index = int(i / 2) if i % 2 == 0 else num_workers - 1 - int(i / 2)
            if i < high_count:
                pod_info[PodType.WORKER][index].set_priority("high")
            else:
                pod_info[PodType.WORKER][index].set_priority("low")
    elif worker_pod_priority in [None, "", "high", "low"]:
        for i in range(num_workers):
            pod_info[PodType.WORKER][i].set_priority(worker_pod_priority)
    else:
        raise ValueError(
            "Not support priority = {}, please set priority = "
            "high/low/a fraction value.".format(worker_pod_priority)
        )
