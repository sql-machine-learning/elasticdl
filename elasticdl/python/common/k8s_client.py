import os
import threading
import traceback

from kubernetes import client, config, watch
from kubernetes.client import (
    V1EnvVar,
    V1EnvVarSource,
    V1ObjectFieldSelector,
    V1PersistentVolumeClaimVolumeSource,
)

from elasticdl.python.common.k8s_resource import parse as parse_resource
from elasticdl.python.common.k8s_volume import parse as parse_volume
from elasticdl.python.common.log_util import default_logger as logger
from elasticdl.python.common.model_helper import load_module

ELASTICDL_APP_NAME = "elasticdl"
ELASTICDL_JOB_KEY = "elasticdl-job-name"
ELASTICDL_REPLICA_TYPE_KEY = "elasticdl-replica-type"
ELASTICDL_REPLICA_INDEX_KEY = "elasticdl-replica-index"


class Client(object):
    def __init__(
        self,
        *,
        image_name,
        namespace,
        job_name,
        event_callback,
        cluster_spec=""
    ):
        """
        ElasticDL k8s client.

        Args:
            image_name: Docker image path for ElasticDL pod.
            namespace: The name of the Kubernetes namespace where ElasticDL
                pods will be created.
            job_name: ElasticDL job name, should be unique in the namespace.
                Used as pod name prefix and value for "elasticdl" label.
            event_callback: If not None, an event watcher will be created and
                events passed to the callback.
        """
        if os.getenv("KUBERNETES_SERVICE_HOST"):
            # We are running inside k8s
            config.load_incluster_config()
        else:
            # Use user's kube config
            config.load_kube_config()

        self.client = client.CoreV1Api()
        self.namespace = namespace
        self.job_name = job_name
        self._image_name = image_name
        self._event_cb = event_callback
        if self._event_cb:
            threading.Thread(
                target=self._watch, name="event_watcher", daemon=True
            ).start()
        self.cluster = None
        if cluster_spec:
            cluster_spec_module = load_module(cluster_spec)
            self.cluster = cluster_spec_module.cluster

    def _watch(self):
        stream = watch.Watch().stream(
            self.client.list_namespaced_pod,
            self.namespace,
            label_selector=ELASTICDL_JOB_KEY + "=" + self.job_name,
        )
        for event in stream:
            try:
                self._event_cb(event)
            except Exception:
                traceback.print_exc()

    def get_master_pod_name(self):
        return "elasticdl-%s-master" % self.job_name

    def get_worker_pod_name(self, worker_id):
        return "elasticdl-%s-worker-%s" % (self.job_name, str(worker_id))

    def get_embedding_service_pod_name(self, embedding_service_id):
        return "elasticdl-%s-embedding-service-%s" % (
            self.job_name,
            str(embedding_service_id),
        )

    def patch_labels_to_pod(self, pod_name, labels_dict):
        body = {"metadata": {"labels": labels_dict}}
        try:
            return self.client.patch_namespaced_pod(
                name=pod_name, namespace=self.namespace, body=body
            )
        except client.api_client.ApiException as e:
            logger.warning("Exception when patching labels to pod: %s\n" % e)
            return None

    def get_master_pod(self):
        try:
            return self.client.read_namespaced_pod(
                name=self.get_master_pod_name(), namespace=self.namespace
            )
        except client.api_client.ApiException as e:
            logger.warning("Exception when reading master pod: %s\n" % e)
            return None

    def get_worker_pod(self, worker_id):
        try:
            return self.client.read_namespaced_pod(
                name=self.get_worker_pod_name(worker_id),
                namespace=self.namespace,
            )
        except client.api_client.ApiException as e:
            logger.warning("Exception when reading worker pod: %s\n" % e)
            return None

    def get_embedding_service_pod(self, embedding_service_id):
        try:
            return self.client.read_namespaced_pod(
                name=self.get_embedding_service_pod_name(embedding_service_id),
                namespace=self.namespace,
            )
        except client.api_client.ApiException as e:
            logger.warning(
                "Exception when reading embedding service pod: %s\n" % e
            )
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

    def _create_pod(self, **kargs):
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
        )

        # Mount data path
        if kargs["volume"]:
            volume_dict = parse_volume(kargs["volume"])
            volume_name = kargs["pod_name"] + "-volume"
            volume = client.V1Volume(
                name=volume_name,
                persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
                    claim_name=volume_dict["claim_name"], read_only=False
                ),
            )
            spec.volumes = [volume]
            container.volume_mounts = [
                client.V1VolumeMount(
                    name=volume_name, mount_path=volume_dict["mount_path"]
                )
            ]

        pod = client.V1Pod(
            spec=spec,
            metadata=client.V1ObjectMeta(
                name=kargs["pod_name"],
                labels={
                    "app": ELASTICDL_APP_NAME,
                    ELASTICDL_JOB_KEY: kargs["job_name"],
                },
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
        env = [
            V1EnvVar(
                name="MY_POD_IP",
                value_from=V1EnvVarSource(
                    field_ref=V1ObjectFieldSelector(field_path="status.podIP")
                ),
            )
        ]
        for key in kargs["envs"]:
            env.append(V1EnvVar(
                name=key,
                value=kargs["envs"][key],
            ))

        pod = self._create_pod(
            pod_name=self.get_master_pod_name(),
            job_name=self.job_name,
            image_name=self._image_name,
            command=["python"],
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
        resp = self.client.create_namespaced_pod(self.namespace, pod)
        logger.info("Master launched. status='%s'" % str(resp.status))

    def _create_worker_pod(self, pod_name, type_key, **kargs):
        # Find that master pod that will be used as the owner reference
        # for this worker pod.
        master_pod = self.get_master_pod()
        pod = self._create_pod(
            pod_name=pod_name,
            job_name=self.job_name,
            image_name=self._image_name,
            command=kargs["command"],
            resource_requests=kargs["resource_requests"],
            resource_limits=kargs["resource_limits"],
            container_args=kargs["args"],
            pod_priority=kargs["pod_priority"],
            image_pull_policy=kargs["image_pull_policy"],
            restart_policy=kargs["restart_policy"],
            volume=kargs["volume"],
            owner_pod=master_pod,
            env=kargs["envs"],
        )
        # Add replica type and index
        pod.metadata.labels[ELASTICDL_REPLICA_TYPE_KEY] = type_key
        pod.metadata.labels[ELASTICDL_REPLICA_INDEX_KEY] = str(
            kargs["worker_id"]
        )
        return self.client.create_namespaced_pod(self.namespace, pod)

    def create_worker(self, **kargs):
        pod_name = self.get_worker_pod_name(kargs["worker_id"])
        return self._create_worker_pod(pod_name, "worker", **kargs)

    def create_embedding_service(self, **kargs):
        pod_name = self.get_embedding_service_pod_name(kargs["worker_id"])
        return self._create_worker_pod(pod_name, "embedding_service", **kargs)

    def delete_master(self):
        logger.info("pod name is %s" % self.get_master_pod_name())
        self.client.delete_namespaced_pod(
            self.get_master_pod_name(),
            self.namespace,
            body=client.V1DeleteOptions(grace_period_seconds=0),
        )

    def delete_worker(self, worker_id):
        self.client.delete_namespaced_pod(
            self.get_worker_pod_name(worker_id),
            self.namespace,
            body=client.V1DeleteOptions(grace_period_seconds=0),
        )

    def delete_embedding_service(self, embedding_service_id):
        self.client.delete_namespaced_pod(
            self.get_embedding_service_pod_name(embedding_service_id),
            self.namespace,
            body=client.V1DeleteOptions(grace_period_seconds=0),
        )
