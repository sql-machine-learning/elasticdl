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

import os
import random
import time
import unittest
from unittest.mock import MagicMock, call

from elasticdl.python.master.k8s_instance_manager import InstanceManager
from elasticdl.python.master.task_dispatcher import _TaskDispatcher


class InstanceManagerTest(unittest.TestCase):
    @unittest.skipIf(
        os.environ.get("K8S_TESTS", "True") == "False",
        "No Kubernetes cluster available",
    )
    def test_create_delete_worker_pod(self):
        task_d = _TaskDispatcher({"f": (0, 10)}, {}, {}, 1, 1)
        task_d.recover_tasks = MagicMock()
        instance_manager = InstanceManager(
            task_d,
            job_name="test-create-worker-pod-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            image_name="gcr.io/google-samples/hello-app:1.0",
            worker_command=["/bin/bash"],
            worker_args=["-c", "echo"],
            namespace="default",
            num_workers=3,
        )

        instance_manager.start_workers()
        max_check_num = 20
        for _ in range(max_check_num):
            time.sleep(3)
            counters = instance_manager.get_worker_counter()
            if counters["Succeeded"] == 3:
                break

        instance_manager.stop_relaunch_and_remove_workers()
        for _ in range(max_check_num):
            time.sleep(3)
            counters = instance_manager.get_worker_counter()
            if not counters:
                break
        task_d.recover_tasks.assert_has_calls(
            [call(0), call(1), call(2)], any_order=True
        )

    @unittest.skipIf(
        os.environ.get("K8S_TESTS", "True") == "False",
        "No Kubernetes cluster available",
    )
    def test_get_worker_addrs(self):
        task_d = _TaskDispatcher({"f": (0, 10)}, {}, {}, 1, 1)
        instance_manager = InstanceManager(
            task_d,
            job_name="test-create-worker-pod-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            image_name="gcr.io/google-samples/hello-app:1.0",
            worker_command=["/bin/bash"],
            worker_args=["-c", "echo"],
            namespace="default",
            num_workers=3,
        )

        instance_manager.start_workers()
        max_check_num = 20
        for _ in range(max_check_num):
            time.sleep(3)
            counters = instance_manager.get_worker_counter()
            if counters["Succeeded"] == 3:
                break

        worker_addrs = instance_manager._get_alive_worker_service_addr()
        self.assertEqual(len(worker_addrs), 3)
        instance_manager.stop_relaunch_and_remove_workers()

    @unittest.skipIf(
        os.environ.get("K8S_TESTS", "True") == "False",
        "No Kubernetes cluster available",
    )
    def test_failed_worker_pod(self):
        """
        Start a pod running a python program destined to fail with
        restart_policy="Never" to test failed_worker_count
        """
        task_d = _TaskDispatcher({"f": (0, 10)}, {}, {}, 1, 1)
        task_d.recover_tasks = MagicMock()
        instance_manager = InstanceManager(
            task_d,
            job_name="test-failed-worker-pod-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            image_name="gcr.io/google-samples/hello-app:1.0",
            worker_command=["/bin/bash"],
            worker_args=["-c", "badcommand"],
            namespace="default",
            num_workers=3,
            restart_policy="Never",
        )
        instance_manager.start_workers()
        max_check_num = 20
        for _ in range(max_check_num):
            time.sleep(3)
            counters = instance_manager.get_worker_counter()
            if counters["Failed"] == 3:
                break

        instance_manager.stop_relaunch_and_remove_workers()
        for _ in range(max_check_num):
            time.sleep(3)
            counters = instance_manager.get_worker_counter()
            if not counters:
                break
        task_d.recover_tasks.assert_has_calls(
            [call(0), call(1), call(2)], any_order=True
        )

    @unittest.skipIf(
        os.environ.get("K8S_TESTS", "True") == "False",
        "No Kubernetes cluster available",
    )
    def test_relaunch_worker_pod(self):
        num_workers = 3
        task_d = _TaskDispatcher({"f": (0, 10)}, {}, {}, 1, 1)
        instance_manager = InstanceManager(
            task_d,
            job_name="test-relaunch-worker-pod-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            image_name="gcr.io/google-samples/hello-app:1.0",
            worker_command=["/bin/bash"],
            worker_args=["-c", "sleep 10"],
            namespace="default",
            num_workers=num_workers,
        )

        instance_manager.start_workers()

        max_check_num = 60
        for _ in range(max_check_num):
            time.sleep(1)
            counters = instance_manager.get_worker_counter()
            if counters["Running"] + counters["Pending"] > 0:
                break
        # Note: There is a slight chance of race condition.
        # Hack to find a worker to remove
        current_workers = set()
        live_workers = set()
        with instance_manager._lock:
            for k, (_, phase) in instance_manager._worker_pods_phase.items():
                current_workers.add(k)
                if phase in ["Running", "Pending"]:
                    live_workers.add(k)
        self.assertTrue(live_workers)

        instance_manager._remove_worker(live_workers.pop())
        # verify a new worker get launched
        found = False
        for _ in range(max_check_num):
            if found:
                break
            time.sleep(1)
            with instance_manager._lock:
                for k in instance_manager._worker_pods_phase:
                    if k not in range(num_workers, num_workers * 2):
                        found = True
        else:
            self.fail("Failed to find newly launched worker.")

        instance_manager.stop_relaunch_and_remove_workers()

    @unittest.skipIf(
        os.environ.get("K8S_TESTS", "True") == "False",
        "No Kubernetes cluster available",
    )
    def test_relaunch_ps_pod(self):
        num_ps = 3
        instance_manager = InstanceManager(
            task_d=None,
            job_name="test-relaunch-ps-pod-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            image_name="gcr.io/google-samples/hello-app:1.0",
            ps_command=["/bin/bash"],
            ps_args=["-c", "sleep 10"],
            namespace="default",
            num_ps=num_ps,
        )

        instance_manager.start_parameter_servers()

        # Check we also have ps services started
        for i in range(num_ps):
            service = instance_manager._k8s_client.get_ps_service(i)
            self.assertTrue(service.metadata.owner_references)
            owner = service.metadata.owner_references[0]
            self.assertEqual(owner.kind, "Pod")
            self.assertEqual(
                owner.name, instance_manager._k8s_client.get_ps_pod_name(i)
            )

        max_check_num = 60
        for _ in range(max_check_num):
            time.sleep(1)
            counters = instance_manager.get_ps_counter()
            if counters["Running"] + counters["Pending"] > 0:
                break
        # Note: There is a slight chance of race condition.
        # Hack to find a ps to remove
        all_current_ps = set()
        all_live_ps = set()
        with instance_manager._lock:
            for k, (_, phase) in instance_manager._ps_pods_phase.items():
                all_current_ps.add(k)
                if phase in ["Running", "Pending"]:
                    all_live_ps.add(k)
        self.assertTrue(all_live_ps)

        ps_to_be_removed = all_live_ps.pop()
        all_current_ps.remove(ps_to_be_removed)
        instance_manager._remove_ps(ps_to_be_removed)
        # Verify a new ps gets launched
        found = False
        for _ in range(max_check_num):
            if found:
                break
            time.sleep(1)
            with instance_manager._lock:
                for k in instance_manager._ps_pods_phase:
                    if k not in all_current_ps:
                        found = True
        else:
            self.fail("Failed to find newly launched ps.")

        instance_manager.stop_relaunch_and_remove_all_ps()


if __name__ == "__main__":
    unittest.main()
