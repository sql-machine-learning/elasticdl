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

from elasticdl.python.common.k8s_client import PodType
from elasticdl.python.master.pod_info import (
    ResourceConfig,
    TypedPodConfig,
    init_pod_info,
    set_critical_pod,
    set_worker_pod_priority,
)


class PodInfoTest(unittest.TestCase):
    def test_pod_resource_config(self):
        resource_config = ResourceConfig(
            resource_requests="cpu=1,memory=4096Mi",
            resource_limits="",
            priority="low",
            image_name="default",
        )
        resource_config.update_memory(1000)
        self.assertEqual(
            resource_config.resource_requests, "cpu=1,memory=1000Mi"
        )
        self.assertEqual(resource_config.resource_limits, "")

    def test_typed_pod_config(self):
        typed_pod_config = TypedPodConfig()
        typed_pod_config.add_typed_pod_config(
            PodType.WORKER,
            2,
            "cpu=1,memory=2048Mi",
            "cpu=1,memory=2048Mi",
            "high",
            "default",
        )
        typed_pod_config.add_typed_pod_config(
            PodType.PS,
            1,
            "cpu=1,memory=2048Mi",
            "cpu=1,memory=2048Mi",
            "high",
            "default",
        )

        self.assertEqual(typed_pod_config.get_typed_pod_num(PodType.WORKER), 2)
        self.assertEqual(typed_pod_config.get_typed_pod_num(PodType.PS), 1)
        self.assertEqual(
            typed_pod_config.get_typed_pod_num(PodType.EVALUATOR), 0
        )
        self.assertListEqual(
            typed_pod_config.get_pod_types(), [PodType.WORKER, PodType.PS]
        )

        worker_resource = typed_pod_config.get_typed_resource_config(
            PodType.WORKER
        )
        self.assertEqual(
            worker_resource.resource_requests, "cpu=1,memory=2048Mi"
        )
        self.assertEqual(worker_resource.priority, "high")

    def test_pod_info(self):
        typed_pod_config = TypedPodConfig()
        typed_pod_config.add_typed_pod_config(
            PodType.WORKER,
            3,
            "cpu=1,memory=2048Mi",
            "cpu=1,memory=2048Mi",
            "high",
            "default",
        )
        typed_pod_config.add_typed_pod_config(
            PodType.EVALUATOR,
            1,
            "cpu=1,memory=2048Mi",
            "cpu=1,memory=2048Mi",
            "high",
            "default",
        )
        typed_pod_config.add_typed_pod_config(
            PodType.CHIEF,
            1,
            "cpu=1,memory=2048Mi",
            "cpu=1,memory=2048Mi",
            "high",
            "default",
        )
        typed_pod_info = init_pod_info(0, typed_pod_config)
        self.assertListEqual(
            list(typed_pod_info[PodType.WORKER].keys()), [0, 1, 2]
        )

        worker0_info = typed_pod_info[PodType.WORKER][0]
        worker1_info = typed_pod_info[PodType.WORKER][1]
        self.assertFalse(worker0_info.is_critical_pod)
        set_critical_pod(typed_pod_info, critical_worker_index={0: 1})
        self.assertTrue(worker0_info.is_critical_pod)
        self.assertEqual(worker0_info.max_relaunch_count, 1)
        self.assertFalse(worker1_info.is_critical_pod)
        self.assertEqual(worker1_info.max_relaunch_count, 0)
        self.assertTrue(typed_pod_info[PodType.EVALUATOR][0].is_critical_pod)
        self.assertTrue(typed_pod_info[PodType.CHIEF][0].is_critical_pod)

        set_worker_pod_priority(typed_pod_info, "0.1")
        self.assertEqual(worker0_info.resource_config.priority, "high")
        self.assertEqual(worker1_info.resource_config.priority, "low")

        max_memory = 8192
        worker0_info.increment_memory(max_memory)
        self.assertEqual(
            worker0_info.resource_config.get_memory_mi(), max_memory
        )
