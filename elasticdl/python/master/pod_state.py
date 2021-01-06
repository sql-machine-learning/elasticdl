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


from collections import namedtuple

from elasticdl.python.common.constants import PodStatus

PodStateFlow = namedtuple(
    "PodStateFlow",
    ("from_status", "to_status", "event_type", "phase", "should_relaunch"),
)

"""
The DAG for the state machine is in the issue
https://github.com/sql-machine-learning/elasticdl/issues/2395#issue-753964852
"""
POD_STATE_FLOWS = [
    PodStateFlow(
        from_status=PodStatus.INITIAL,
        to_status=PodStatus.PENDING,
        event_type=["ADDED", "MODIFIED"],
        phase="Pending",
        should_relaunch=False,
    ),
    PodStateFlow(
        from_status=PodStatus.INITIAL,
        to_status=PodStatus.RUNNING,
        event_type=["ADDED", "MODIFIED"],
        phase="Running",
        should_relaunch=False,
    ),
    PodStateFlow(
        from_status=PodStatus.PENDING,
        to_status=PodStatus.RUNNING,
        event_type=["ADDED", "MODIFIED"],
        phase="Running",
        should_relaunch=False,
    ),
    PodStateFlow(
        from_status=PodStatus.PENDING,
        to_status=PodStatus.SUCCEEDED,
        event_type=["ADDED", "MODIFIED"],
        phase="Succeeded",
        should_relaunch=False,
    ),
    PodStateFlow(
        from_status=PodStatus.PENDING,
        to_status=PodStatus.FAILED,
        event_type=["ADDED", "MODIFIED"],
        phase="Failed",
        should_relaunch=True,
    ),
    PodStateFlow(
        from_status=PodStatus.RUNNING,
        to_status=PodStatus.SUCCEEDED,
        event_type=["ADDED", "MODIFIED"],
        phase="Succeeded",
        should_relaunch=False,
    ),
    PodStateFlow(
        from_status=PodStatus.RUNNING,
        to_status=PodStatus.FAILED,
        event_type=["ADDED", "MODIFIED"],
        phase="Failed",
        should_relaunch=True,
    ),
    PodStateFlow(
        from_status=PodStatus.PENDING,
        to_status=PodStatus.DELETED,
        event_type=["DELETED"],
        phase=None,
        should_relaunch=True,
    ),
    PodStateFlow(
        from_status=PodStatus.RUNNING,
        to_status=PodStatus.DELETED,
        event_type=["DELETED"],
        phase=None,
        should_relaunch=True,
    ),
    PodStateFlow(
        from_status=PodStatus.SUCCEEDED,
        to_status=PodStatus.DELETED,
        event_type=["DELETED"],
        phase=None,
        should_relaunch=False,
    ),
    PodStateFlow(
        from_status=PodStatus.FAILED,
        to_status=PodStatus.DELETED,
        event_type=["DELETED"],
        phase=None,
        should_relaunch=False,
    ),
]


def get_pod_state_flow(from_status, event_type, phase):
    for pod_state_flow in POD_STATE_FLOWS:
        if (
            from_status == pod_state_flow.from_status
            and event_type in pod_state_flow.event_type
            and (pod_state_flow.phase is None or phase == pod_state_flow.phase)
        ):
            return pod_state_flow

    return None
