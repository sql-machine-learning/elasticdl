import logging
import os
import unittest
import time
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

from unittest.mock import MagicMock, call
from elasticdl.master.k8s_worker_manager import WorkerManager
from elasticdl.master.task_queue import _TaskQueue


class WorkerManagerTest(unittest.TestCase):
    @unittest.skipIf(os.environ.get('K8S_TESTS', 'False') == 'True', 'No Kubernetes cluster available')
    def testCreateDeleteWorkerPod(self):
        task_q = _TaskQueue({"f": 10}, 1, 1)
        task_q.recover_tasks = MagicMock()
        worker_servicer = WorkerManager(
            task_q,
            job_name="test-create-worker-pod",
            #worker_image="elasticdl:dev",
            worker_image="gcr.io/google-samples/hello-app:1.0",
            command=["/bin/ls"],
            args=[],
            namespace="default",
            num_worker=3
        )
        worker_servicer.start_workers()
        counters = worker_servicer.get_counters()
        max_check_num = 20
        for _ in range(max_check_num):
            time.sleep(3)
            counters = worker_servicer.get_counters()
            print(counters)
            if counters["Succeeded"] == 3:
                break

        worker_servicer.remove_workers()
        for _ in range(max_check_num):
            time.sleep(3)
            counters = worker_servicer.get_counters()
            print(counters)
            if not counters:
                break
        task_q.recover_tasks.assert_has_calls(
            [call(0), call(1), call(2)], any_order=True
        )

    @unittest.skipIf(os.environ.get('K8S_TESTS', 'False') == 'True', 'No Kubernetes cluster available')
    def testFailedWorkerPod(self):
        """
        Start a pod running a python program destined to fail with restart_policy="Never"
        to test failed_worker_count
        """
        task_q = _TaskQueue({"f": 10}, 1, 1)
        task_q.recover_tasks = MagicMock()
        worker_servicer = WorkerManager(
            task_q,
            job_name="test-create-worker-pod",
            worker_image="gcr.io/google-samples/hello-app:1.0",
            command=["badcommand"],
            args=["badargs"],
            namespace="default",
            num_worker=3
        )
        worker_servicer.start_workers(restart_policy="Never")
        max_check_num = 20
        for _ in range(max_check_num):
            time.sleep(3)
            counters = worker_servicer.get_counters()
            print(counters)
            if counters["Failed"] == 3:
                break

        worker_servicer.remove_workers()
        for _ in range(max_check_num):
            time.sleep(3)
            counters = worker_servicer.get_counters()
            print(counters)
            if not counters:
                break
        task_q.recover_tasks.assert_has_calls(
            [call(0), call(1), call(2)], any_order=True
        )

if __name__ == '__main__':
    unittest.main()
