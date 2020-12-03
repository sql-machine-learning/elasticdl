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

from elasticdl.python.common.constants import PodStatus
from elasticdl.python.common.k8s_client import PodType
from elasticdl.python.master.pod_event_callbacks import TaskRescheduleCallback
from elasticdl.python.master.pod_manager import PodManager
from elasticdl.python.tests.test_utils import create_task_manager


class PodManagerTest(unittest.TestCase):
    @unittest.skipIf(
        os.environ.get("K8S_TESTS", "True") == "False",
        "No Kubernetes cluster available",
    )
    def test_create_delete_worker_pod(self):
        pod_manager = PodManager(
            job_name="test-create-worker-pod-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            image_name="ubuntu:18.04",
            namespace="default",
            num_workers=2,
            envs=[],
        )
        pod_manager.set_up(
            worker_command=["/bin/bash"], worker_args=["-c", "echo"],
        )
        pod_manager._k8s_client.start_watch_events()

        pod_manager.start_workers()
        max_check_num = 20
        for _ in range(max_check_num):
            time.sleep(3)
            counters = pod_manager.get_pod_counter(pod_type=PodType.WORKER)
            print("Counters: {}".format(counters))
            if counters[PodStatus.SUCCEEDED] == 2:
                break

        pod_manager._not_created_worker_id = [2]
        pod_manager._worker_pod_priority[2] = None
        pod_manager._process_worker()
        for _ in range(max_check_num):
            time.sleep(3)
            counters = pod_manager.get_pod_counter(pod_type=PodType.WORKER)
            if counters[PodStatus.SUCCEEDED] == 3:
                break

        pod_manager.stop_relaunch_and_remove_pods(pod_type=PodType.WORKER)
        for _ in range(max_check_num):
            time.sleep(3)
            counters = pod_manager.get_pod_counter(pod_type=PodType.WORKER)
            if counters[PodStatus.DELETED] == 3:
                break
        else:
            self.fail("Cannot get expected 3 deleted pods.")

    @unittest.skipIf(
        os.environ.get("K8S_TESTS", "True") == "False",
        "No Kubernetes cluster available",
    )
    def test_get_worker_addrs(self):
        pod_manager = PodManager(
            job_name="test-create-worker-pod-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            image_name="ubuntu:18.04",
            namespace="default",
            num_workers=3,
            envs=[],
        )
        pod_manager.set_up(
            worker_command=["/bin/bash"], worker_args=["-c", "sleep 5 #"],
        )
        pod_manager._k8s_client.start_watch_events()

        pod_manager.start_workers()
        max_check_num = 20
        for _ in range(max_check_num):
            time.sleep(3)
            counters = pod_manager.get_pod_counter(pod_type=PodType.WORKER)
            if counters[PodStatus.RUNNING]:
                worker_addrs = pod_manager.get_alive_worker_name_addr()
                self.assertEqual(
                    len(worker_addrs), counters[PodStatus.RUNNING]
                )

        pod_manager.stop_relaunch_and_remove_pods(pod_type=PodType.WORKER)

    @unittest.skipIf(
        os.environ.get("K8S_TESTS", "True") == "False",
        "No Kubernetes cluster available",
    )
    def test_failed_worker_pod(self):
        """
        Start a pod running a python program destined to fail with
        restart_policy="Never" to test failed_worker_count
        """
        task_manager = create_task_manager({"f": (0, 10)}, {})
        task_manager.recover_tasks = MagicMock()
        pod_manager = PodManager(
            job_name="test-failed-worker-pod-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            image_name="ubuntu:18.04",
            namespace="default",
            num_workers=3,
            restart_policy="Never",
            envs=[],
        )
        pod_manager.set_up(
            worker_command=["/bin/bash"], worker_args=["-c", "badcommand"],
        )
        pod_manager.add_pod_event_callback(
            TaskRescheduleCallback(task_manager=task_manager)
        )
        pod_manager._k8s_client.start_watch_events()
        pod_manager.start_workers()
        max_check_num = 20
        for _ in range(max_check_num):
            time.sleep(3)
            counters = pod_manager.get_pod_counter(pod_type=PodType.WORKER)
            print("Counters: {}".format(counters))
            if counters[PodStatus.FAILED] == 3:
                break

        pod_manager.stop_relaunch_and_remove_pods(pod_type=PodType.WORKER)
        for _ in range(max_check_num):
            time.sleep(3)
            counters = pod_manager.get_pod_counter(pod_type=PodType.WORKER)
            if counters[PodStatus.DELETED] == 3:
                break
        else:
            self.fail("Cannot get 3 deleted worker pods as expected.")
        task_manager.recover_tasks.assert_has_calls(
            [call(0), call(1), call(2)], any_order=True
        )

    @unittest.skipIf(
        os.environ.get("K8S_TESTS", "True") == "False",
        "No Kubernetes cluster available",
    )
    def test_relaunch_worker_pod(self):
        num_workers = 3
        pod_manager = PodManager(
            job_name="test-relaunch-worker-pod-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            image_name="ubuntu:18.04",
            namespace="default",
            num_workers=num_workers,
            envs=[],
        )
        pod_manager.set_up(
            worker_command=["/bin/bash"], worker_args=["-c", "sleep 10 #"],
        )
        pod_manager._k8s_client.start_watch_events()
        pod_manager.start_workers()

        max_check_num = 60
        for _ in range(max_check_num):
            time.sleep(1)
            counters = pod_manager.get_pod_counter(pod_type=PodType.WORKER)
            if counters[PodStatus.RUNNING] + counters[PodStatus.PENDING] > 0:
                break
        # Note: There is a slight chance of race condition.
        # Hack to find a worker to remove
        alive_workers = pod_manager.get_pod_infos(
            PodType.WORKER, [PodStatus.RUNNING, PodStatus.PENDING]
        )
        self.assertTrue(alive_workers)

        pod_manager._remove_worker(alive_workers.pop().id)
        # verify a new worker get launched
        for _ in range(max_check_num):
            current_alive_workers = pod_manager.get_pod_infos(
                PodType.WORKER, [PodStatus.RUNNING, PodStatus.PENDING]
            )
            print("Current alive workers: {}".format(current_alive_workers))
            # The former worker id is from 0 ~ num_workers - 1
            # If a new worker is launched, the worker id is >= num_workers
            new_launched_workers = [
                pod_info
                for pod_info in current_alive_workers
                if pod_info.id >= num_workers
            ]
            if new_launched_workers:
                break
        else:
            self.fail("Cannot to find any newly launched worker.")

        pod_manager.stop_relaunch_and_remove_pods(pod_type=PodType.WORKER)

    @unittest.skipIf(
        os.environ.get("K8S_TESTS", "True") == "False",
        "No Kubernetes cluster available",
    )
    def test_launch_ps_pod(self):
        num_ps = 3
        pod_manager = PodManager(
            job_name="test-relaunch-ps-pod-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            image_name="ubuntu:18.04",
            namespace="default",
            num_ps=num_ps,
        )
        pod_manager.set_up(
            ps_command=["/bin/bash"], ps_args=["-c", "sleep 10 #"],
        )
        pod_manager._k8s_client.start_watch_events()
        pod_manager.start_parameter_servers()

        # Check we also have ps services started
        for i in range(num_ps):
            service = pod_manager._k8s_client.get_ps_service(i)
            self.assertTrue(service.metadata.owner_references)
            owner = service.metadata.owner_references[0]
            self.assertEqual(owner.kind, "Pod")
            self.assertEqual(
                owner.name, pod_manager._k8s_client.get_ps_pod_name(i)
            )

        max_check_num = 60
        for _ in range(max_check_num):
            time.sleep(1)
            counters = pod_manager.get_pod_counter(pod_type=PodType.PS)
            if counters[PodStatus.RUNNING] + counters[PodStatus.PENDING] > 0:
                break
        else:
            self.fail("PS pod cannot start within the time limit.")

        pod_manager.stop_relaunch_and_remove_pods(pod_type=PodType.PS)


if __name__ == "__main__":
    unittest.main()
