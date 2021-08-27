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

import base64
import unittest

from kubernetes import client

from elasticdl_client.common.k8s_client import ClusterSpec, PodType

test_spec = """
{
   "pod_spec": {
      "labels": {
         "elasticdl.org/app-name": "elasticdl",
         "elasticdl.org/site": "hangzhou"
      },
      "annotations": {
         "tag.elasticdl.org/optimization": "enabled",
         "master-config": "{\\"cpu\\": 0.4}"
      },
      "affinity": {
         "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
               "nodeSelectorTerms": [
                  {
                     "matchExpressions": [
                        {
                           "key": "elasticdl.org/logic-pool",
                           "operator": "In",
                           "values": [
                              "ElasticDL"
                           ]
                        }
                     ]
                  }
               ]
            }
         }
      },
      "tolerations": [
         {
            "effect": "NoSchedule",
            "key": "elasticdl.org/logic-pool",
            "operator": "Equal",
            "value": "ElasticDL"
         }
      ],
      "env": [
         {
            "name": "LOG_ENABLED",
            "value": "true"
         }
      ]
   },
   "service_spec": {
      "clusterIP": "None",
      "type": "ClusterIP"
   },
   "edl-master_pod_spec": {
      "labels": {
         "elasticdl.org/xyz": "Sun"
      }
   },
   "worker_pod_spec": {
      "labels": {
         "elasticdl.org/xyz": "Earth"
      }
   },
   "ps_pod_spec": {
      "labels": {
         "elasticdl.org/xyz": "Moon"
      }
   },
   "master_pod_spec": {
      "labels": {
         "elasticdl.org/xyz": "Mars"
      }
   }
}
"""


def create_test_pod(name="test"):
    container = client.V1Container(
        name=name + "_pod_name", image=name + "_image_name", command="bash"
    )

    # Pod
    spec = client.V1PodSpec(containers=[container])

    pod = client.V1Pod(
        spec=spec,
        metadata=client.V1ObjectMeta(
            name=name + "_pod_name", namespace="elasticdl"
        ),
    )
    return pod


def create_test_service(name="test_service"):
    metadata = client.V1ObjectMeta(name=name)
    spec = client.V1ServiceSpec()

    service = client.V1Service(
        api_version="v1", kind="Service", metadata=metadata, spec=spec
    )
    return service


class ClusterSpecTest(unittest.TestCase):
    def validate_cluster_spec(self, cluster_spec_config):
        cluster_spec = ClusterSpec(cluster_spec_json=cluster_spec_config)
        pod = create_test_pod("test_spec")
        pod = cluster_spec.patch_pod(pod, "other")

        self.assertEqual(
            pod.metadata.labels["elasticdl.org/app-name"], "elasticdl"
        )
        self.assertEqual(pod.metadata.labels["elasticdl.org/site"], "hangzhou")
        self.assertEqual(
            pod.metadata.annotations["tag.elasticdl.org/optimization"],
            "enabled",
        )
        expected_tolerations = [
            client.V1Toleration(
                effect="NoSchedule",
                key="elasticdl.org/logic-pool",
                operator="Equal",
                value="ElasticDL",
            )
        ]
        self.assertEqual(pod.spec.tolerations, expected_tolerations)
        match_expressions = [
            client.V1NodeSelectorRequirement(
                key="elasticdl.org/logic-pool",
                operator="In",
                values=["ElasticDL"],
            )
        ]

        expected_affinity = client.V1Affinity(
            node_affinity=client.V1NodeAffinity(
                required_during_scheduling_ignored_during_execution=(
                    client.V1NodeSelector(
                        node_selector_terms=[
                            client.V1NodeSelectorTerm(
                                match_expressions=match_expressions
                            )
                        ]
                    )
                )
            )
        )
        self.assertEqual(pod.spec.affinity, expected_affinity)

        expected_env = []
        expected_env.append(client.V1EnvVar(name="LOG_ENABLED", value="true"))
        self.assertEqual(pod.spec.containers[0].env, expected_env)

        pod = create_test_pod("test_spec")
        pod = cluster_spec.patch_pod(pod, PodType.MASTER)
        self.assertEqual(pod.metadata.labels["elasticdl.org/xyz"], "Sun")

        pod = create_test_pod("test_spec")
        pod = cluster_spec.patch_pod(pod, PodType.WORKER)
        self.assertEqual(pod.metadata.labels["elasticdl.org/xyz"], "Earth")

        pod = create_test_pod("test_spec")
        pod = cluster_spec.patch_pod(pod, PodType.PS)
        self.assertEqual(pod.metadata.labels["elasticdl.org/xyz"], "Moon")

        pod = create_test_pod("test_spec")
        pod = cluster_spec.patch_pod(pod, PodType.CHIEF)
        self.assertEqual(pod.metadata.labels["elasticdl.org/xyz"], "Mars")

    def test_pod_spec_json(self):
        self.validate_cluster_spec(cluster_spec_config=test_spec)

    def test_pod_spec_base64(self):
        spec_base64_bytes = base64.b64encode(test_spec.encode("utf-8"))
        self.validate_cluster_spec(spec_base64_bytes)

    def test_service_spec(self):
        cluster_spec = ClusterSpec(cluster_spec_json=test_spec)
        service = create_test_service("test_spec")
        service = cluster_spec.patch_service(service)
        self.assertEqual(service.spec.type, "ClusterIP")
        self.assertEqual(service.spec.cluster_ip, "None")
