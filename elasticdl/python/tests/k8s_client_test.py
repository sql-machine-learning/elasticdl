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

import json
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

        # Start 2 ps pods
        for i in range(2):
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
        while tracker._count < 6:
            time.sleep(1)

        # Check ps pods labels
        for i in range(2):
            ps = c.get_ps_pod(i)
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
            c.create_ps_service(i)

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
            c.create_worker_service(i)

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
        c.patch_worker_service(0, 3)
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

    def test_get_tf_config_data(self):
        c = k8s.Client(
            image_name="gcr.io/google-samples/hello-app:1.0",
            namespace="default",
            job_name="test-job-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
        )

        tf_config_cluster = "'cluster': \
          {'ps': \
              ['elasticdl-JOBNAME-ps-0.NAMESPACE.svc:PSPORT', \
              'elasticdl-JOBNAME-ps-1.NAMESPACE.svc:PSPORT'], \
           'worker': \
              ['elasticdl-JOBNAME-worker-0.NAMESPACE.svc:WORKERPORT', \
              'elasticdl-JOBNAME-worker-1.NAMESPACE-ps-1.svc:WORKERPORT'] \
           } "
        tf_config_cluster = tf_config_cluster.replace("JOBNAME", c.job_name)
        tf_config_cluster = tf_config_cluster.replace("NAMESPACE", c.namespace)
        tf_config_cluster = tf_config_cluster.replace(
            "PSPORT", str(k8s._PS_SERVICE_PORT)
        )
        tf_config_cluster = tf_config_cluster.replace(
            "WORKERPORT", str(k8s._WORKER_SERVICE_PORT)
        )

        tf_config_cluster_dict = json.loads(tf_config_cluster)

        ps0_config = c.get_tf_config_data(2, 2, k8s.PodType.PS, 0)
        ps0_config_dict = json.loads(ps0_config)
        self.assertEqual(tf_config_cluster_dict, ps0_config_dict["cluster"])
        self.assertEqual(ps0_config_dict["task"]["type"], "ps")
        self.assertEqual(ps0_config_dict["task"]["index"], 0)

        worker1_config = c.get_tf_config_data(2, 2, k8s.PodType.WORKER, 1)
        worker1_config_dict = json.loads(worker1_config)
        self.assertEqual(
            tf_config_cluster_dict, worker1_config_dict["cluster"]
        )
        self.assertEqual(worker1_config_dict["task"]["type"], "worker")
        self.assertEqual(worker1_config_dict["task"]["index"], 1)

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
