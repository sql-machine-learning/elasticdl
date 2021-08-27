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

from elasticdl.python.common import k8s_client as k8s
from elasticdl.python.common.k8s_client import PodType


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
        c.start_watch_events()

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
            _ = c.create_typed_pod(
                PodType.WORKER,
                str(i),
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
            worker = c.get_typed_pod(PodType.WORKER, i)
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

        # Start 2 ps pods
        for i in range(2):
            _ = c.create_typed_pod(
                PodType.PS,
                i,
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
        while tracker._count < 6:
            time.sleep(1)

        # Check ps pods labels
        for i in range(2):
            ps = c.get_typed_pod(PodType.PS, i)
            self.assertEqual(
                ps.metadata.labels[k8s.ELASTICDL_JOB_KEY], c.job_name
            )
            self.assertEqual(
                ps.metadata.labels[k8s.ELASTICDL_REPLICA_TYPE_KEY], "ps"
            )
            self.assertEqual(
                ps.metadata.labels[k8s.ELASTICDL_REPLICA_INDEX_KEY], str(i)
            )

        # Start 2 ps services
        for i in range(2):
            c.create_service(PodType.PS, i)

        # Check ps services
        for i in range(2):
            service = c.get_ps_service(i)
            self.assertIsNotNone(service)
            self.assertEqual(
                service.spec.selector[k8s.ELASTICDL_JOB_KEY], c.job_name
            )
            self.assertEqual(
                service.spec.selector[k8s.ELASTICDL_REPLICA_TYPE_KEY], "ps"
            )
            self.assertEqual(
                service.spec.selector[k8s.ELASTICDL_REPLICA_INDEX_KEY], str(i)
            )

        # Start 2 worker services
        for i in range(2):
            c.create_service(PodType.WORKER, i)

        # Check worker services
        for i in range(2):
            service = c.get_worker_service(i)
            self.assertIsNotNone(service)
            self.assertEqual(
                service.spec.selector[k8s.ELASTICDL_JOB_KEY], c.job_name
            )
            self.assertEqual(
                service.spec.selector[k8s.ELASTICDL_REPLICA_TYPE_KEY], "worker"
            )
            self.assertEqual(
                service.spec.selector[k8s.ELASTICDL_REPLICA_INDEX_KEY], str(i)
            )

        # patch worker service
        c.patch_service(PodType.WORKER, 0, 3)
        service = c.get_worker_service(0)
        self.assertIsNotNone(service)
        self.assertEqual(
            service.spec.selector[k8s.ELASTICDL_JOB_KEY], c.job_name
        )
        self.assertEqual(
            service.spec.selector[k8s.ELASTICDL_REPLICA_TYPE_KEY], "worker"
        )
        self.assertEqual(
            service.spec.selector[k8s.ELASTICDL_REPLICA_INDEX_KEY], str(3)
        )

        # Delete master and all ps and workers should also be deleted
        c.delete_master()

        # wait for all ps, workers and services to be deleted
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
        c.start_watch_events()

        # Start 1 worker
        resource = "cpu=100m,memory=64M"
        worker_id = 1
        worker_pod = c.create_typed_pod(
            PodType.WORKER,
            worker_id,
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
        c.delete_typed_pod(PodType.WORKER, worker_id)

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
