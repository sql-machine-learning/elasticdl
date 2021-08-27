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

import argparse
import copy
import json
import os
import random
import time
import unittest
from time import sleep
from unittest.mock import MagicMock, call

from elasticai_api.common.constants import WorkerEnv
from elasticdl.python.common.constants import PodStatus
from elasticdl.python.common.k8s_client import (
    _PS_SERVICE_PORT,
    _WORKER_SERVICE_PORT,
    PodType,
)
from elasticdl.python.master.pod_event_callbacks import TaskRescheduleCallback
from elasticdl.python.master.pod_info import PodInfo, TypedPodConfig
from elasticdl.python.master.pod_manager import (
    PodManager,
    build_environment_variables,
    get_critical_worker_index,
    is_huge_memory,
    should_launch_worker_after_ps_running,
)
from elasticdl.python.tests.test_utils import create_task_manager
from elasticdl_client.common.constants import DistributionStrategy

RESOURCE = "cpu=1,memory=8192Mi"


class PodManagerTest(unittest.TestCase):
    def setUp(self):
        self._typed_pod_config = TypedPodConfig()
        self._typed_pod_config.add_typed_pod_config(
            PodType.WORKER, 2, RESOURCE, RESOURCE, None, None
        )
        self._typed_pod_config.add_typed_pod_config(
            PodType.PS, 3, RESOURCE, RESOURCE, None, None
        )

    @unittest.skipIf(
        os.environ.get("K8S_TESTS", "True") == "False",
        "No Kubernetes cluster available",
    )
    def test_create_delete_worker_pod(self):
        pod_manager = PodManager(
            typed_pod_config=self._typed_pod_config,
            job_name="test-create-worker-pod-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            image_name="ubuntu:18.04",
            namespace="default",
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
            if counters[PodStatus.SUCCEEDED] == 2:
                break

        pod_manager._not_created_workers = [(PodType.WORKER, 2)]
        pod_manager.pod_info[PodType.WORKER][2].original_index = 1
        pod_manager._process_worker()
        for _ in range(max_check_num):
            time.sleep(3)
            counters = pod_manager.get_pod_counter(pod_type=PodType.WORKER)
            if counters[PodStatus.SUCCEEDED] == 3:
                break

        pod_manager.stop_relaunch_and_remove_all_pods()
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
            typed_pod_config=self._typed_pod_config,
            job_name="test-create-worker-pod-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            image_name="ubuntu:18.04",
            namespace="default",
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
                worker_addrs = pod_manager.get_alive_worker_id_addr()
                self.assertEqual(
                    len(worker_addrs), counters[PodStatus.RUNNING]
                )

        pod_manager.stop_relaunch_and_remove_all_pods()

    @unittest.skipIf(
        os.environ.get("K8S_TESTS", "True") == "False",
        "No Kubernetes cluster available",
    )
    def test_failed_worker_pod(self):
        """
        Start a pod running a python program destined to fail with
        restart_policy="Never" to test failed_worker_count
        """
        task_manager = create_task_manager([("f", 0, 10)], [])
        task_manager.recover_tasks = MagicMock()
        self._typed_pod_config._typed_pod_num[PodType.WORKER] = 3
        pod_manager = PodManager(
            typed_pod_config=self._typed_pod_config,
            job_name="test-failed-worker-pod-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            image_name="ubuntu:18.04",
            namespace="default",
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
            if counters[PodStatus.FAILED] == 3:
                break

        pod_manager.stop_relaunch_and_remove_all_pods()
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
        self._typed_pod_config._typed_pod_num[PodType.WORKER] = 3
        pod_manager = PodManager(
            typed_pod_config=self._typed_pod_config,
            job_name="test-relaunch-worker-pod-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            image_name="ubuntu:18.04",
            namespace="default",
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

        pod_manager.remove_worker(alive_workers.pop().id)
        # verify a new worker get launched
        for _ in range(max_check_num):
            current_alive_workers = pod_manager.get_pod_infos(
                PodType.WORKER, [PodStatus.RUNNING, PodStatus.PENDING]
            )
            # The former worker id is from 0 ~ num_workers - 1
            # If a new worker is launched, the worker id is >= num_workers
            new_launched_workers = [
                pod_info
                for pod_info in current_alive_workers
                if pod_info.id >= 3
            ]
            if new_launched_workers:
                break
            sleep(1)
        else:
            self.fail("Cannot to find any newly launched worker.")

        pod_manager.stop_relaunch_and_remove_all_pods()

    @unittest.skipIf(
        os.environ.get("K8S_TESTS", "True") == "False",
        "No Kubernetes cluster available",
    )
    def test_launch_ps_pod(self):
        num_ps = 3
        self._typed_pod_config._typed_pod_num[PodType.PS] = num_ps
        pod_manager = PodManager(
            typed_pod_config=self._typed_pod_config,
            job_name="test-relaunch-ps-pod-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            image_name="ubuntu:18.04",
            namespace="default",
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
                owner.name, pod_manager._k8s_client.get_pod_name(PodType.PS, i)
            )

        max_check_num = 60
        for _ in range(max_check_num):
            time.sleep(1)
            counters = pod_manager.get_pod_counter(pod_type=PodType.PS)
            if counters[PodStatus.RUNNING] + counters[PodStatus.PENDING] > 0:
                break
        else:
            self.fail("PS pod cannot start within the time limit.")

        pod_manager.stop_relaunch_and_remove_all_pods()

    def test_all_worker_exited(self):
        typed_pod_config = copy.deepcopy(self._typed_pod_config)
        typed_pod_config.add_typed_pod_config(
            PodType.WORKER, 3, RESOURCE, RESOURCE, None, None
        )
        typed_pod_config.add_typed_pod_config(
            PodType.EVALUATOR, 1, RESOURCE, RESOURCE, None, None
        )
        pod_manager = PodManager(
            typed_pod_config=typed_pod_config,
            job_name="test-failed-worker-pod-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            image_name="ubuntu:18.04",
            namespace="default",
            restart_policy="Never",
            envs=[],
        )
        pod_info = PodInfo(
            pod_type=PodType.WORKER,
            id=0,
            name="ut-worker-0",
            pod_ip="0.0.0.1",
            status=PodStatus.FAILED,
            start_time=None,
        )
        pod_manager.pod_info[PodType.WORKER][pod_info.id] = pod_info
        self.assertTrue(pod_manager.all_workers_failed)
        self.assertTrue(pod_manager.all_workers_and_evaluators_exited)

    def test_get_tf_config_data(self):
        job_name = "test-tf_config-%d-%d" % (
            int(time.time()),
            random.randint(1, 101),
        )
        namespace = "default"
        typed_pod_config = copy.deepcopy(self._typed_pod_config)
        typed_pod_config.add_typed_pod_config(
            PodType.EVALUATOR, 1, RESOURCE, RESOURCE, None, None,
        )
        typed_pod_config.add_typed_pod_config(
            PodType.PS, 2, RESOURCE, RESOURCE, None, None
        )
        typed_pod_config.add_typed_pod_config(
            PodType.CHIEF, 1, RESOURCE, RESOURCE, None, None
        )
        pod_manager = PodManager(
            typed_pod_config=typed_pod_config,
            job_name=job_name,
            image_name="ubuntu:18.04",
            namespace=namespace,
        )

        tf_config_cluster = '{"cluster": \
            {"ps": \
              ["JOBNAME-edljob-ps-0:PSPORT", \
               "JOBNAME-edljob-ps-1:PSPORT"], \
            "worker": \
              ["JOBNAME-edljob-worker-0:WORKERPORT", \
               "JOBNAME-edljob-worker-1:WORKERPORT"], \
            "evaluator": \
              ["JOBNAME-edljob-evaluator-0:WORKERPORT"], \
            "master": \
              ["JOBNAME-edljob-master-0:WORKERPORT"] \
            }}'
        tf_config_cluster = tf_config_cluster.replace("JOBNAME", job_name)
        tf_config_cluster = tf_config_cluster.replace("NAMESPACE", namespace)
        tf_config_cluster = tf_config_cluster.replace(
            "PSPORT", str(_PS_SERVICE_PORT)
        )
        tf_config_cluster = tf_config_cluster.replace(
            "WORKERPORT", str(_WORKER_SERVICE_PORT)
        )

        tf_config_cluster_dict = json.loads(tf_config_cluster)

        ps0_config = pod_manager.get_tf_config_data(PodType.PS, 0)
        self.assertEqual(
            tf_config_cluster_dict["cluster"], ps0_config["cluster"]
        )
        self.assertEqual(ps0_config["task"]["type"], "ps")
        self.assertEqual(ps0_config["task"]["index"], 0)

        worker1_config = pod_manager.get_tf_config_data(PodType.WORKER, 1)
        self.assertEqual(
            tf_config_cluster_dict["cluster"], worker1_config["cluster"]
        )
        self.assertEqual(worker1_config["task"]["type"], "worker")
        self.assertEqual(worker1_config["task"]["index"], 1)

        chief_worker_config = pod_manager.get_tf_config_data(
            PodType.CHIEF, 0
        )
        self.assertEqual(
            tf_config_cluster_dict["cluster"], chief_worker_config["cluster"]
        )
        self.assertEqual(chief_worker_config["task"]["type"], "master")
        self.assertEqual(chief_worker_config["task"]["index"], 0)

    def test_build_environment_variables(self):
        os.environ["ELASTICDL_abc"] = "abc"
        args = argparse.Namespace(
            envs="a=1,b=2",
            num_workers=2,
            port=50001,
            populate_env_names="ELASTICDL_.*",
        )
        envs = build_environment_variables(args)
        env_dict = {env.name: env.value for env in envs}
        self.assertTrue("a" in env_dict)
        self.assertTrue("b" in env_dict)
        self.assertTrue(WorkerEnv.MASTER_ADDR in env_dict)
        self.assertTrue(WorkerEnv.WORKER_NUM in env_dict)
        self.assertTrue("ELASTICDL_abc" in env_dict)

    def test_critical_worker_index(self):
        args = argparse.Namespace(
            need_elasticdl_job_service=False,
            need_tf_config=True,
            distribution_strategy=DistributionStrategy.PARAMETER_SERVER,
            critical_worker_index="default",
            num_evaluators=0,
            num_workers=3,
            relaunch_on_worker_failure=1,
        )
        critical_index = get_critical_worker_index(True, args)
        self.assertEqual(critical_index, {0: 0})

        args = argparse.Namespace(
            need_elasticdl_job_service=False,
            need_tf_config=True,
            distribution_strategy=DistributionStrategy.ALLREDUCE,
            critical_worker_index="default",
        )
        critical_index = get_critical_worker_index(False, args)
        self.assertEqual(critical_index, {})

        args = argparse.Namespace(
            need_elasticdl_job_service=False,
            need_tf_config=True,
            distribution_strategy=DistributionStrategy.PARAMETER_SERVER,
            critical_worker_index="none",
            num_evaluators=1,
            num_workers=3,
            relaunch_on_worker_failure=1,
        )
        critical_index = get_critical_worker_index(True, args)
        self.assertEqual(critical_index, {})

        args = argparse.Namespace(
            need_elasticdl_job_service=False,
            need_tf_config=True,
            distribution_strategy=DistributionStrategy.ALLREDUCE,
            critical_worker_index="1:1,3:1,9:1",
            num_evaluators=1,
        )
        critical_index = get_critical_worker_index(True, args)
        self.assertEqual(critical_index, {1: 1, 3: 1, 9: 1})

    def test_should_launch_worker_after_ps_running(self):
        args = argparse.Namespace(
            launch_worker_after_ps_running="default",
            num_ps_pods=1,
            ps_resource_request="cpu=20,memory=4096Mi",
        )
        is_tfv1_ps = True

        self.assertTrue(
            should_launch_worker_after_ps_running(is_tfv1_ps, True, args)
        )

        args = argparse.Namespace(
            launch_worker_after_ps_running="default",
            num_ps_pods=1,
            ps_resource_request="cpu=2,memory=4096Mi",
        )
        self.assertFalse(
            should_launch_worker_after_ps_running(is_tfv1_ps, False, args)
        )

        args = argparse.Namespace(
            launch_worker_after_ps_running="on",
            num_ps_pods=1,
            ps_resource_request="cpu=1,memory=4096Mi",
        )
        self.assertTrue(
            should_launch_worker_after_ps_running(is_tfv1_ps, False, args)
        )

    def test_is_huge_memory(self):
        typed_pod_config = TypedPodConfig()
        resource = "cpu=1,memory=20480Mi"
        typed_pod_config.add_typed_pod_config(
            PodType.WORKER, 6, resource, resource, "", ""
        )
        self.assertTrue(is_huge_memory(typed_pod_config))

        typed_pod_config = TypedPodConfig()
        resource = "cpu=1,memory=204800Mi"
        typed_pod_config.add_typed_pod_config(
            PodType.WORKER, 1, resource, resource, "", ""
        )
        self.assertTrue(is_huge_memory(typed_pod_config))

        typed_pod_config = TypedPodConfig()
        resource = "cpu=1,memory=20480Mi"
        typed_pod_config.add_typed_pod_config(
            PodType.WORKER, 3, resource, resource, "", ""
        )
        self.assertFalse(is_huge_memory(typed_pod_config))

    def test_get_recommended_memory(self):
        pod_manager = PodManager(
            typed_pod_config=self._typed_pod_config,
            job_name="test-create-worker-pod-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            image_name="ubuntu:18.04",
            namespace="default",
            envs=[],
        )
        pod_manager.wait_chief_worker_execution = False
        mem = pod_manager._get_recommended_memory(PodType.WORKER)
        self.assertEqual(mem, 0.0)

        pod_manager._worker_resource_monitor._worker_memory = 1
        mem = pod_manager._get_recommended_memory(PodType.WORKER)
        self.assertEqual(mem, 4096)

        pod_manager._worker_resource_monitor._worker_memory = 3000
        mem = pod_manager._get_recommended_memory(PodType.WORKER)
        self.assertEqual(mem, 5400)


if __name__ == "__main__":
    unittest.main()
