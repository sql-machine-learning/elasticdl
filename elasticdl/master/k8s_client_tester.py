"""
To run the tester on local machine, in this directory:
    python k8s_client_tester.py
To run this tester inside a k8s cluster:
    TODO: create image
    TODO: create yaml
    kubectl apply -f tester.yaml
"""

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


def main():
    tracker = WorkerTracker()

    c = k8s.Client(
        worker_image="gcr.io/google-samples/hello-app:1.0",
        namespace="default",
        job_name="training-job",
        master_addr="",
        event_callback=tracker.event_cb,
    )

    # Start 3 workers
    for i in range(3):
        c.create_worker("worker-%d" % i)
        time.sleep(5)

    # wait for workers to be added
    while tracker._count < 3:
        time.sleep(1)
    
    # delete all workers
    for i in range(tracker._count): 
        c.delete_worker("worker-%d" % i)

    # wait for workers to be deleted
    while tracker._count > 3:
        time.sleep(1)

if __name__ == "__main__":
    main()
