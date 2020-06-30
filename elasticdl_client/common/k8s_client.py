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
import traceback

import yaml
from kubernetes import client, config
from kubernetes.client import V1EnvVar, V1EnvVarSource, V1ObjectFieldSelector

from elasticdl_client.common.k8s_resource import parse as parse_resource
from elasticdl_client.common.k8s_volume import parse_volume_and_mount
from elasticdl_client.common.log_utils import default_logger as logger
from elasticdl_client.common.module_utils import load_module

ELASTICDL_APP_NAME = "elasticdl"
ELASTICDL_JOB_KEY = "elasticdl-job-name"
ELASTICDL_REPLICA_TYPE_KEY = "elasticdl-replica-type"
ELASTICDL_REPLICA_INDEX_KEY = "elasticdl-replica-index"
_FTLIB_GOSSIP_CONTAINER_PORT = 7946


def get_master_pod_name(job_name):
    return "elasticdl-%s-master" % job_name


def append_pod_ip_to_env(env):
    pod_ip_var = V1EnvVar(
        name="MY_POD_IP",
        value_from=V1EnvVarSource(
            field_ref=V1ObjectFieldSelector(field_path="status.podIP")
        ),
    )
    if env:
        env.append(pod_ip_var)
    else:
        env = [pod_ip_var]
    return env


class Client(object):
    def __init__(
        self,
        *,
        image_name,
        namespace,
        job_name,
        cluster_spec="",
        force_use_kube_config_file=False
    ):
        """
        ElasticDL k8s client.

        Args:
            image_name: Docker image path for ElasticDL pod.
            namespace: The name of the Kubernetes namespace where ElasticDL
                pods will be created.
            job_name: ElasticDL job name, should be unique in the namespace.
                Used as pod name prefix and value for "elasticdl" label.
            force_use_kube_config_file: If true, force to load the cluster
                config from ~/.kube/config. Otherwise, if it's in a process
                running in a K8S environment, it loads the incluster config,
                if not, it loads the kube config file.
        """
        try:
            if (
                os.getenv("KUBERNETES_SERVICE_HOST")
                and not force_use_kube_config_file
            ):
                # We are running inside a k8s cluster
                config.load_incluster_config()
                logger.info("Load the incluster config.")
            else:
                # Use user's kube config
                config.load_kube_config()
                logger.info("Load the kube config file.")
        except Exception as ex:
            traceback.print_exc()
            raise Exception(
                "Failed to load configuration for Kubernetes:\n%s" % str(ex)
            )

        self.client = client.CoreV1Api()
        self.namespace = namespace
        self.job_name = job_name
        self._image_name = image_name
        self.cluster = None
        if cluster_spec:
            cluster_spec_module = load_module(cluster_spec)
            self.cluster = cluster_spec_module.cluster

    def get_master_pod_name(self):
        return get_master_pod_name(self.job_name)

    def patch_labels_to_pod(self, pod_name, labels_dict):
        body = {"metadata": {"labels": labels_dict}}
        try:
            return self.client.patch_namespaced_pod(
                name=pod_name, namespace=self.namespace, body=body
            )
        except client.api_client.ApiException as e:
            logger.warning("Exception when patching labels to pod: %s\n" % e)
            return None

    @staticmethod
    def create_owner_reference(owner_pod):
        owner_ref = (
            [
                client.V1OwnerReference(
                    api_version="v1",
                    block_owner_deletion=True,
                    kind="Pod",
                    name=owner_pod.metadata.name,
                    uid=owner_pod.metadata.uid,
                )
            ]
            if owner_pod
            else None
        )
        return owner_ref

    def create_pod(self, **kargs):
        # Container
        pod_resource_requests = kargs["resource_requests"]
        pod_resource_limits = kargs["resource_limits"]
        pod_resource_limits = (
            pod_resource_limits
            if pod_resource_limits
            else pod_resource_requests
        )
        ports = (
            [
                client.V1ContainerPort(
                    container_port=_FTLIB_GOSSIP_CONTAINER_PORT, name="gossip"
                ),
            ]
            if "expose_ports" in kargs and kargs["expose_ports"]
            else None
        )
        container = client.V1Container(
            name=kargs["pod_name"],
            image=kargs["image_name"],
            command=kargs["command"],
            resources=client.V1ResourceRequirements(
                requests=parse_resource(pod_resource_requests),
                limits=parse_resource(pod_resource_limits),
            ),
            args=kargs["container_args"],
            image_pull_policy=kargs["image_pull_policy"],
            env=kargs["env"],
            ports=ports,
        )

        # Pod
        spec = client.V1PodSpec(
            containers=[container],
            restart_policy=kargs["restart_policy"],
            priority_class_name=kargs["pod_priority"],
            termination_grace_period_seconds=kargs.get(
                "termination_period", None
            ),
        )

        # Mount data path
        if kargs["volume"]:
            volumes, volume_mounts = parse_volume_and_mount(
                kargs["volume"], kargs["pod_name"]
            )
            spec.volumes = volumes
            container.volume_mounts = volume_mounts

        pod = client.V1Pod(
            spec=spec,
            metadata=client.V1ObjectMeta(
                name=kargs["pod_name"],
                labels=self._get_common_labels(),
                owner_references=self.create_owner_reference(
                    kargs["owner_pod"]
                ),
                namespace=self.namespace,
            ),
        )
        if self.cluster:
            pod = self.cluster.with_pod(pod)

        return pod

    def create_master(self, **kargs):
        pod = self._create_master_pod_obj(**kargs)
        self.client.create_namespaced_pod(self.namespace, pod)
        logger.info("Master launched.")

    def dump_master_yaml(self, **kargs):
        pod = self._create_master_pod_obj(**kargs)
        pod_dict = self.client.api_client.sanitize_for_serialization(pod)
        with open(kargs["yaml"], "w") as f:
            yaml.safe_dump(pod_dict, f, default_flow_style=False)

    def _create_master_pod_obj(self, **kargs):
        env = []
        if "envs" in kargs:
            for key in kargs["envs"]:
                env.append(V1EnvVar(name=key, value=kargs["envs"][key]))
        env = append_pod_ip_to_env(env)

        pod = self.create_pod(
            pod_name=self.get_master_pod_name(),
            job_name=self.job_name,
            image_name=self._image_name,
            command=["/bin/bash"],
            resource_requests=kargs["resource_requests"],
            resource_limits=kargs["resource_limits"],
            container_args=kargs["args"],
            pod_priority=kargs["pod_priority"],
            image_pull_policy=kargs["image_pull_policy"],
            restart_policy=kargs["restart_policy"],
            volume=kargs["volume"],
            owner_pod=None,
            env=env,
        )
        # Add replica type and index
        pod.metadata.labels[ELASTICDL_REPLICA_TYPE_KEY] = "master"
        pod.metadata.labels[ELASTICDL_REPLICA_INDEX_KEY] = "0"
        pod.api_version = "v1"
        pod.kind = "Pod"
        return pod

    def delete_master(self):
        logger.info("pod name is %s" % self.get_master_pod_name())
        self.delete_pod(self.get_master_pod_name())

    def delete_pod(self, pod_name):
        self.client.delete_namespaced_pod(
            pod_name,
            self.namespace,
            body=client.V1DeleteOptions(grace_period_seconds=0),
        )

    def _get_common_labels(self):
        """Labels that should be attached to all k8s objects belong to
           current job.
        """
        return {"app": ELASTICDL_APP_NAME, ELASTICDL_JOB_KEY: self.job_name}
