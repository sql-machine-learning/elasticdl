import unittest

import os
import time

from elasticdl.master import k8s

class WorkerTracker(object):
    def __init__(self):
        self._count = 0

    def event_cb(self, event):
        print("----- %s -----\n" % event["type"])
        if event["type"] == "ADDED":
            self._count += 1
        elif event["type"] == "DELETED":
            self._count -= 1


@unittest.skipIf(os.environ.get('K8S_TESTS', 'True') == 'False', 'No Kubernetes cluster available')
class K8sClientTest(unittest.TestCase):
    def test_client(self):
        tracker = WorkerTracker()

        c = k8s.Client(
            worker_image="gcr.io/google-samples/hello-app:1.0",
            namespace="default",
            job_name="training-job",
            event_callback=tracker.event_cb,
        )

        # Start 3 workers
        for i in range(3):
            c.create_worker("worker-%d" % i, "500m", "500m", "64Mi", "64Mi")
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
