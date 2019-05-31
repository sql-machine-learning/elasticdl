import os
import unittest
import time

import tensorflow as tf

tf.enable_eager_execution()

from unittest.mock import MagicMock, call
from elasticdl.python.elasticdl.master.k8s_worker_manager import WorkerManager
from elasticdl.python.elasticdl.master.task_queue import _TaskQueue


class WorkerManagerTest(unittest.TestCase):
    @unittest.skipIf(os.environ.get('K8S_TESTS', 'True') == 'False', 'No Kubernetes cluster available')
    def testCreateDeleteWorkerPod(self):
        task_q = _TaskQueue({"f": 10}, 1, 1)
        task_q.recover_tasks = MagicMock()
        worker_manager = WorkerManager(
            task_q,
            job_name="test-create-worker-pod",
            worker_image="gcr.io/google-samples/hello-app:1.0",
            command=["echo"],
            args=[],
            namespace="default",
            num_worker=3
        )

        worker_manager.start_workers()
        max_check_num = 20
        for _ in range(max_check_num):
            time.sleep(3)
            counters = worker_manager.get_counters()
            print(counters)
            if counters["Succeeded"] == 3:
                break

        worker_manager.stop_relaunch_and_remove_workers()
        for _ in range(max_check_num):
            time.sleep(3)
            counters = worker_manager.get_counters()
            print(counters)
            if not counters:
                break
        task_q.recover_tasks.assert_has_calls(
            [call(0), call(1), call(2)], any_order=True
        )

    @unittest.skipIf(os.environ.get('K8S_TESTS', 'True') == 'False', 'No Kubernetes cluster available')
    def testFailedWorkerPod(self):
        """
        Start a pod running a python program destined to fail with restart_policy="Never"
        to test failed_worker_count
        """
        task_q = _TaskQueue({"f": 10}, 1, 1)
        task_q.recover_tasks = MagicMock()
        worker_manager = WorkerManager(
            task_q,
            job_name="test-create-worker-pod",
            worker_image="gcr.io/google-samples/hello-app:1.0",
            command=["badcommand"],
            args=["badargs"],
            namespace="default",
            num_worker=3,
            restart_policy="Never"
        )
        worker_manager.start_workers()
        max_check_num = 20
        for _ in range(max_check_num):
            time.sleep(3)
            counters = worker_manager.get_counters()
            print(counters)
            if counters["Failed"] == 3:
                break

        worker_manager.stop_relaunch_and_remove_workers()
        for _ in range(max_check_num):
            time.sleep(3)
            counters = worker_manager.get_counters()
            print(counters)
            if not counters:
                break
        task_q.recover_tasks.assert_has_calls(
            [call(0), call(1), call(2)], any_order=True
        )

    @unittest.skipIf(os.environ.get('K8S_TESTS', 'True') == 'False', 'No Kubernetes cluster available')
    def testRelaunchWorkerPod(self):
        task_q = _TaskQueue({"f": 10}, 1, 1)
        worker_manager = WorkerManager(
            task_q,
            job_name="test-relaunch-worker-pod",
            worker_image="gcr.io/google-samples/hello-app:1.0",
            command=["sleep 10"],
            args=[],
            namespace="default",
            num_worker=3
        )

        worker_manager.start_workers()
        
        max_check_num = 60
        for _ in range(max_check_num):
            time.sleep(1)
            counters = worker_manager.get_counters()
            print(counters)
            if counters["Runing"] + counters["Pending"] > 0:
                break
        # Note: There is a slight chance of race condition.
        # Hack to find a worker to remove
        current_workers = set()
        live_workers = set()
        with worker_manager._lock:
            for k, (_, phase) in worker_manager._pods_phase.items():
                current_workers.add(k)
                if phase in ["Running", "Pending"]:
                    live_workers.add(k)
        self.assertTrue(live_workers)

        worker_manager._remove_worker(live_workers.pop())
        # verify a new worker get launched
        found = False
        print(current_workers)
        for _ in range(max_check_num):
            if found:
                break
            time.sleep(1)
            counters = worker_manager.get_counters()
            print(counters)
            with worker_manager._lock:
                for k in worker_manager._pods_phase:
                    if k not in current_workers:
                        found = True
        else:
            self.fail("Failed to find newly launched worker.")
        
        worker_manager.stop_relaunch_and_remove_workers()

if __name__ == '__main__':
    unittest.main()
