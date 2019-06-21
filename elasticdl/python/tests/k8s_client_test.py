import unittest

import os
import random
import time

from elasticdl.python.elasticdl.common import k8s_client as k8s


class WorkerTracker(object):
    def __init__(self):
        self._count = 0

    def event_cb(self, event):
        if event["type"] == "ADDED":
            self._count += 1
        elif event["type"] == "DELETED":
            self._count -= 1


@unittest.skipIf(
    os.environ.get("K8S_TESTS", "True") == "False",
    "No Kubernetes cluster available",
)
class K8sClientTest(unittest.TestCase):
    def test_client(self):
        tracker = WorkerTracker()

        c = k8s.Client(
            image_name="gcr.io/google-samples/hello-app:1.0",
            namespace="default",
            job_name="test-job-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            event_callback=tracker.event_cb,
        )

        # Start 3 workers
        resource = {"cpu": "100m", "memory": "64M"}
        for i in range(3):
            _ = c.create_worker(
                worker_id="worker-%d" % i,
                resource_requests=resource,
                resource_limits=resource,
                command=["echo"],
                priority=None,
                args=None,
                mount_path=None,
                volume_name=None,
                image_pull_policy="Never",
                restart_policy="Never"
            )
            time.sleep(5)

        # wait for workers to be added
        while tracker._count < 3:
            time.sleep(1)

        # delete all workers
        for i in range(tracker._count):
            c.delete_worker("worker-%d" % i)

        # wait for workers to be deleted
        while tracker._count > 0:
            time.sleep(1)


if __name__ == "__main__":
    unittest.main()
