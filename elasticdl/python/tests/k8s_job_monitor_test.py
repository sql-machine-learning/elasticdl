import os
import time
import unittest

from elasticdl.python.common import k8s_client as k8s
from elasticdl.python.common.k8s_job_monitor import EdlJobMonitor, PodMonitor


class WorkerTracker(object):
    def __init__(self):
        self._count = 0

    def event_cb(self, event):
        if event["type"] == "ADDED":
            self._count += 1
        elif event["type"] == "DELETED":
            self._count -= 1


def launch_elasticdl_job(image_name, namespace, job_name, worker_num, ps_num):
    tracker = WorkerTracker()

    c = k8s.Client(
        image_name=image_name,
        namespace=namespace,
        job_name=job_name,
        event_callback=tracker.event_cb,
    )

    # Start master
    resource = "cpu=100m,memory=64M"
    c.create_master(
        resource_requests=resource,
        resource_limits=resource,
        pod_priority=None,
        args=None,
        volume=None,
        image_pull_policy="Never",
        restart_policy="Never",
    )
    pod_count = 1
    while tracker._count < pod_count:
        time.sleep(1)

    # Start 3 workers
    for i in range(worker_num):
        _ = c.create_worker(
            worker_id=str(i),
            resource_requests=resource,
            resource_limits=resource,
            command=["echo"],
            pod_priority=None,
            args=None,
            volume=None,
            image_pull_policy="Never",
            restart_policy="Never",
        )
        time.sleep(5)

    # Wait for workers to be added
    pod_count = pod_count + worker_num
    while tracker._count < pod_count:
        time.sleep(1)

    # Start 2 ps pods
    for i in range(ps_num):
        _ = c.create_ps(
            ps_id=str(i),
            resource_requests=resource,
            resource_limits=resource,
            command=["echo"],
            pod_priority=None,
            args=None,
            volume=None,
            image_pull_policy="Never",
            restart_policy="Never",
        )
        time.sleep(5)

    # Wait for ps to be added
    pod_count = pod_count + ps_num
    while tracker._count < pod_count:
        time.sleep(1)

    # Start 2 ps services
    for i in range(2):
        c.create_ps_service(i)


@unittest.skipIf(
    os.environ.get("K8S_TESTS", "True") == "False",
    "No Kubernetes cluster available",
)
class K8sClientTest(unittest.TestCase):
    def setUp(self):
        self.namespace = "default"
        self.image_name = "gcr.io/google-samples/hello-app:1.0"

    def test_pod_monitor(self):
        tracker = WorkerTracker()

        c = k8s.Client(
            image_name=self.image_name,
            namespace=self.namespace,
            job_name="test-job-%d" % (int(time.time())),
            event_callback=tracker.event_cb,
        )

        # Start master
        resource = "cpu=100m,memory=64M"
        _ = c.create_worker(
            worker_id="0",
            resource_requests=resource,
            resource_limits=resource,
            command=["echo"],
            pod_priority=None,
            args=None,
            volume=None,
            image_pull_policy="Never",
            restart_policy="Never",
        )

        pod_name = c.get_worker_pod_name(0)
        pod_monitor = PodMonitor(
            namespace=self.namespace, pod_name=pod_name
        )
        pod_succeed = pod_monitor.monitor_status()
        self.assertTrue(pod_succeed)

    def test_job_monitor(self):
        namespace = self.namespace
        job_name = "test-job-%d" % (int(time.time()))
        worker_num = 3
        ps_num = 2
        launch_elasticdl_job(
            image_name=self.image_name,
            namespace=namespace,
            job_name=job_name,
            worker_num=worker_num,
            ps_num=ps_num,
        )
        edl_job_monitor = EdlJobMonitor(
            namespace=namespace,
            job_name=job_name,
            worker_num=worker_num,
            ps_num=ps_num,
        )
        job_succeed = edl_job_monitor.monitor_status()
        self.assertTrue(job_succeed)


if __name__ == "__main__":
    unittest.main()
