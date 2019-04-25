import logging
import unittest
import time
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

from k8s_worker_servicer import WorkerServicer


class WorkerServicerTest(unittest.TestCase):
    def testCreateDeleteWorkerPod(self):
        worker_servicer = WorkerServicer(
            job_name="test-create-worker-pod",
            #worker_image="elasticdl:dev",
            worker_image="gcr.io/google-samples/hello-app:1.0",
            command=["/bin/ls"],
            namespace="default",
            worker_num=3
        )
        worker_servicer.start_workers()
        counters = worker_servicer.get_counters()
        max_check_num = 20
        for _ in range(max_check_num):
            time.sleep(3)
            counters = worker_servicer.get_counters()
            if counters["finished_worker_count"] == 3:
                break
        self.assertEqual(counters["finished_worker_count"], 3)
        self.assertEqual(counters["unfinished_worker_count"], 0)
        self.assertEqual(counters["pod_count"], 3)

        worker_servicer.remove_workers()
        for _ in range(max_check_num):
            time.sleep(3)
            counters = worker_servicer.get_counters()
            if counters["pod_count"] == 0:
                break
        self.assertEqual(counters["pod_count"], 0)

    def testFailedWorkerPod(self):
        """
        Start a pod running a python program destined to fail with restart_policy="Never"
        to test failed_worker_count
        """
        worker_servicer = WorkerServicer(
            job_name="test-create-worker-pod",
            worker_image="gcr.io/google-samples/hello-app:1.0",
            command=["badcommand"],
            args=["badargs"],
            namespace="default",
            worker_num=3
        )
        worker_servicer.start_workers(restart_policy="Never")
        counters = worker_servicer.get_counters()
        max_check_num = 20
        for _ in range(max_check_num):
            time.sleep(3)
            counters = worker_servicer.get_counters()
            if counters["finished_worker_count"] + counters["failed_worker_count"] == 3:
                break
        self.assertEqual(counters["finished_worker_count"], 0)
        self.assertEqual(counters["unfinished_worker_count"], 0)
        self.assertEqual(counters["failed_worker_count"], 3)
        self.assertEqual(counters["pod_count"], 3)

        worker_servicer.remove_workers()
        for _ in range(max_check_num):
            time.sleep(3)
            counters = worker_servicer.get_counters()
            if counters["pod_count"] == 0:
                break
        self.assertEqual(counters["pod_count"], 0)

if __name__ == '__main__':
    unittest.main()
