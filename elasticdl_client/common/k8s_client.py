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
import traceback

import six
import yaml
from kubernetes import client, config
from kubernetes.client import V1EnvVar, V1EnvVarSource, V1ObjectFieldSelector

from elasticdl_client.common.constants import ClusterSpecConfig
from elasticdl_client.common.k8s_resource import parse as parse_resource
from elasticdl_client.common.k8s_volume import parse_volume_and_mount
from elasticdl_client.common.log_utils import default_logger as logger
from elasticdl_client.common.module_utils import load_module

ELASTICDL_APP_NAME = "elasticdl"
ELASTICDL_JOB_KEY = "elasticdl-job-name"
ELASTICDL_REPLICA_TYPE_KEY = "elasticdl-replica-type"
ELASTICDL_REPLICA_INDEX_KEY = "elasticdl-replica-index"


def get_master_pod_name(job_name):
    return "elasticdl-%s-master" % job_name


class PodType(object):
    MASTER = "master"
    PS = "ps"
    WORKER = "worker"


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


def try_get_class(module, name):
    try:
        cls = getattr(module, name)
        return cls
    except AttributeError:
        return None


def check_list_get_type(type_name):
    if type_name.startswith("list["):
        return True, type_name.split("[")[1].split("]")[0]
    else:
        return False, None


def get_instance_from_value(type_name, value):
    # Args:
    #   type_name: a class name in client, or a basic type such as string,
    #              int, float, etc.
    #   value: a dict corresponding to the class, or a string/int/float
    #          value if basic type.
    # Return: a class instance if class, the basic type value otherwise.

    cls = try_get_class(client, type_name)
    if not cls:
        # not a class
        is_a_list, list_type = check_list_get_type(type_name)
        if is_a_list:
            value_list = []
            # value must be a list
            for v in value:
                value_list.append(get_instance_from_value(list_type, v))
            return value_list
        else:
            return value
    # value must be a dict
    args = {}
    for attr, attr_type in six.iteritems(cls.openapi_types):
        if cls.attribute_map[attr] in value:
            attr_value = get_instance_from_value(
                attr_type, value[cls.attribute_map[attr]]
            )
            args[attr] = attr_value
    cls_inst = cls(**args)
    return cls_inst


class ClusterSpec(object):
    def __init__(self, cluster_spec_json="", cluster_spec=""):
        """
        Cluster spec for adding on-premise k8s cluster specification,
        including pod and service specifications if needed.

        Args:
            cluster_spec_json: An JSON-encoded string. After decoding, it is
                a dict. This dict may contains:
                (1) "pod_spec" to add pod specifications to all pods;
                (2) "master_spec" to add pod specifications to master pod;
                (3) "ps_spec" to add pod specifications to ps pod;
                (4) "worker_spec" to add pod specifications to worker pod;
                (5) "service_spec" to add service specifications to ps service.
                Supported pod specifications include labels, annotations,
                tolerations, affinity, env.
            cluster_spec: A Python file name. The corresponding file defines a
                cluster class instance, which has `with_pod` and `with_service`
                methods to add pod/service specifications.
        """
        self._cluster = None
        self._cluster_spec = None
        if cluster_spec:
            cluster_spec_module = load_module(cluster_spec)
            self._cluster = cluster_spec_module.cluster
        if cluster_spec_json:
            self._cluster_spec = json.loads(cluster_spec_json)

    def patch_pod(self, pod, pod_type):
        if self._cluster:
            pod = self._cluster.with_pod(pod)
        elif self._cluster_spec:
            if ClusterSpecConfig.POD_SPEC in self._cluster_spec:
                pod = self._patch_pod_with_spec(
                    pod, self._cluster_spec[ClusterSpecConfig.POD_SPEC]
                )
            pod_type_spec_name = pod_type + ClusterSpecConfig.POD_SPEC_SUFFIX
            if pod_type_spec_name in self._cluster_spec:
                pod = self._patch_pod_with_spec(
                    pod, self._cluster_spec[pod_type_spec_name]
                )
        return pod

    def patch_service(self, service):
        if self._cluster:
            service = self._cluster.with_service(service)
        if (
            self._cluster_spec
            and ClusterSpecConfig.SERVICE_SPEC in self._cluster_spec
        ):
            service = self._patch_service_with_spec(
                service, self._cluster_spec[ClusterSpecConfig.SERVICE_SPEC]
            )
        return service

    def _patch_pod_with_spec(self, pod, spec):
        # Add labels if any
        if "labels" in spec:
            labels = spec["labels"]
            if not pod.metadata.labels:
                pod.metadata.labels = {}
            for label_name in labels:
                pod.metadata.labels[label_name] = labels[label_name]

        # Add annotations if any
        if "annotations" in spec:
            annotations = spec["annotations"]
            if not pod.metadata.annotations:
                pod.metadata.annotations = {}
            for annotation_name in annotations:
                pod.metadata.annotations[annotation_name] = annotations[
                    annotation_name
                ]

        # Add affinity if any
        if "affinity" in spec:
            pod.spec.affinity = get_instance_from_value(
                "V1Affinity", spec["affinity"]
            )

        # Add tolerations if any
        if "tolerations" in spec:
            tolerations = spec["tolerations"]
            if not pod.spec.tolerations:
                pod.spec.tolerations = []
            for toleration in tolerations:
                pod.spec.tolerations.append(
                    get_instance_from_value("V1Toleration", toleration)
                )

        # Add env if any
        if "env" in spec:
            for container in pod.spec.containers:
                if not container.env:
                    container.env = []
                for env in spec["env"]:
                    container.env.append(
                        get_instance_from_value("V1EnvVar", env)
                    )

        return pod

    def _patch_service_with_spec(self, service, spec):
        for attr, attr_type in six.iteritems(
            client.V1ServiceSpec.openapi_types
        ):
            if client.V1ServiceSpec.attribute_map[attr] in spec:
                attr_value = get_instance_from_value(
                    attr_type, spec[client.V1ServiceSpec.attribute_map[attr]]
                )
                setattr(service.spec, attr, attr_value)
        return service


class Client(object):
    def __init__(
        self,
        *,
        image_name,
        namespace,
        job_name,
        cluster_spec="",
        cluster_spec_json="",
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
        self.cluster_spec = ClusterSpec(cluster_spec_json, cluster_spec)

    def get_master_pod_name(self):
        return get_master_pod_name(self.job_name)

    def patch_labels_to_pod(self, pod_name, labels_dict):
        body = {"metadata": {"labels": labels_dict}}
        try:
            return self.client.patch_namespaced_pod(
                name=pod_name, namespace=self.namespace, body=body
            )
        except client.ApiException as e:
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
        pod = self.cluster_spec.patch_pod(pod, kargs["pod_type"])

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
            pod_type=PodType.MASTER,
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
