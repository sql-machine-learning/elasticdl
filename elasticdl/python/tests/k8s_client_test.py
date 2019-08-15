import os
import random
import time
import unittest

from elasticdl.python.common import k8s_client as k8s


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
        while tracker._count < 1:
            time.sleep(1)

        # Check master pod labels
        master = c.get_master_pod()
        self.assertEqual(
            master.metadata.labels[k8s.ELASTICDL_JOB_KEY], c.job_name
        )
        self.assertEqual(
            master.metadata.labels[k8s.ELASTICDL_REPLICA_TYPE_KEY], "master"
        )
        self.assertEqual(
            master.metadata.labels[k8s.ELASTICDL_REPLICA_INDEX_KEY], "0"
        )

        # Start 3 workers
        for i in range(3):
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
        while tracker._count < 4:
            time.sleep(1)

        # Check worker pods labels
        for i in range(3):
            worker = c.get_worker_pod(i)
            self.assertEqual(
                worker.metadata.labels[k8s.ELASTICDL_JOB_KEY], c.job_name
            )
            self.assertEqual(
                worker.metadata.labels[k8s.ELASTICDL_REPLICA_TYPE_KEY],
                "worker",
            )
            self.assertEqual(
                worker.metadata.labels[k8s.ELASTICDL_REPLICA_INDEX_KEY], str(i)
            )

        # Start 3 embedding services
        for i in range(3):
            _ = c.create_embedding_service(
                embedding_service_id=str(i),
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

        # Wait for embedding services to be added
        while tracker._count < 7:
            time.sleep(1)

        # Check embedding services pods labels
        for i in range(3):
            embedding_service = c.get_embedding_service_pod(i)
            self.assertEqual(
                embedding_service.metadata.labels[k8s.ELASTICDL_JOB_KEY], c.job_name
            )
            self.assertEqual(
                embedding_service.metadata.labels[k8s.ELASTICDL_REPLICA_TYPE_KEY],
                "embedding_service",
            )
            self.assertEqual(
                embedding_service.metadata.labels[k8s.ELASTICDL_REPLICA_INDEX_KEY], str(i)
            )

        # Delete master and all workers should also be deleted
        c.delete_master()

        # wait for workers to be deleted
        while tracker._count > 0:
            time.sleep(1)

    def test_patch_labels_to_pod(self):
        tracker = WorkerTracker()

        c = k8s.Client(
            image_name="gcr.io/google-samples/hello-app:1.0",
            namespace="default",
            job_name="test-job-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            event_callback=tracker.event_cb,
        )

        # Start 1 worker
        resource = "cpu=100m,memory=64M"
        worker_name = "worker-1"
        worker_pod = c.create_worker(
            worker_id=worker_name,
            resource_requests=resource,
            resource_limits=resource,
            command=["echo"],
            pod_priority=None,
            args=None,
            volume=None,
            image_pull_policy="Never",
            restart_policy="Never",
        )

        label_k = "status"
        label_v = "finished"
        modified_pod = c.patch_labels_to_pod(
            pod_name=worker_pod.metadata.name, labels_dict={label_k: label_v}
        )

        # Wait for the worker to be added
        while tracker._count != 1:
            time.sleep(1)

        # Patching labels to an existing pod should work correctly
        self.assertEqual(modified_pod.metadata.labels[label_k], label_v)

        # Delete the worker
        c.delete_worker(worker_name)

        # Wait for the worker to be deleted
        while tracker._count == 1:
            time.sleep(1)

        # Patching a non-existent pod should return None
        modified_pod = c.patch_labels_to_pod(
            pod_name=worker_pod.metadata.name, labels_dict={label_k: label_v}
        )
        self.assertEqual(modified_pod, None)


if __name__ == "__main__":
    unittest.main()
