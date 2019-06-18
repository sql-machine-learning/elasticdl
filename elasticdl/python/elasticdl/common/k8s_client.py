import logging
import os
import threading
import traceback

from kubernetes import client, config, watch
from kubernetes.client import (
    V1PersistentVolumeClaimVolumeSource as pvcVolumeSource,
    V1EnvVar,
    V1EnvVarSource,
    V1ObjectFieldSelector,
)
from elasticdl.python.elasticdl.common.k8s_utils import parse_resource


class Client(object):
    def __init__(self, *, image_name, namespace, job_name, event_callback):
        """
        ElasticDL k8s client.

        Args:
            image_name: Docker image path for ElasticDL pod.
            namespace: k8s namespace for ElasticDL pods.
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

        self._v1 = client.CoreV1Api()
        self._logger = logging.getLogger(__name__)
        self._image = image_name
        self._ns = namespace
        self._job_name = job_name
        self._event_cb = event_callback
        if self._event_cb:
            threading.Thread(
                target=self._watch, name="event_watcher", daemon=True
            ).start()

    def _watch(self):
        self._logger.info(self._ns)
        stream = watch.Watch().stream(
            self._v1.list_namespaced_pod,
            self._ns,
            label_selector="elasticdl_job_name=" + self._job_name,
        )
        for event in stream:
            try:
                self._event_cb(event)
            except Exception:
                traceback.print_exc()

    def get_master_pod_name(self):
        return "elasticdl-%s-master" % self._job_name

    def get_worker_pod_name(self, worker_id):
        return "elasticdl-%s-worker-%s" % (self._job_name, str(worker_id))

    @staticmethod
    def _create_pod(
        pod_name,
        job_name,
        image_name,
        command,
        resource_request,
        resource_limit,
        container_args,
        pod_priority,
        image_pull_policy,
        restart_policy,
        volume_name,
        mount_path,
        owner_pod,
        env,
    ):
        # Container
        resource_requests = {
            "cpu": resource_request["cpu"],
            "memory": resource_request["memory"],
        }
        resource_limits = {
            "cpu": resource_limit["cpu"],
            "memory": resource_limit["memory"],
        }
        container = client.V1Container(
            name=pod_name,
            image=image_name,
            command=command,
            resources=client.V1ResourceRequirements(
                requests=resource_requests, limits=resource_limits
            ),
            args=container_args,
        )
        if image_pull_policy is not None:
            container.image_pull_policy = image_pull_policy

        if env is not None:
            container.env = env

        # Pod
        spec = client.V1PodSpec(
            containers=[container], restart_policy=restart_policy
        )

        # Mount data path
        if volume_name is not None and mount_path is not None:
            volume = client.V1Volume(
                name=volume_name,
                persistent_volume_claim=pvcVolumeSource(
                    claim_name="fileserver-claim", read_only=False
                ),
            )
            spec.volumes = [volume]
            container.volume_mounts = [
                client.V1VolumeMount(name=volume_name, mount_path=mount_path)
            ]

        if pod_priority is not None:
            spec.priority_class_name = pod_priority

        owner_ref = (
            [
                client.V1OwnerReference(
                    api_version="v1",
                    block_owner_deletion=True,
                    kind="Pod",
                    name=owner_pod[0].metadata.name,
                    uid=owner_pod[0].metadata.uid,
                )
            ]
            if owner_pod is not None and len(owner_pod) != 0
            else None
        )

        pod = client.V1Pod(
            spec=spec,
            metadata=client.V1ObjectMeta(
                name=pod_name,
                labels={"app": "elasticdl", "elasticdl_job_name": job_name},
                owner_references=owner_ref,
            ),
        )
        return pod

    def create_master(
        self,
        job_name,
        image_name,
        model_file,
        master_resource_request,
        master_resource_limit,
        worker_resource_request,
        worker_resource_limit,
        master_pod_priority,
        image_pull_policy,
        volume_name,
        mount_path,
        restart_policy,
        args,
    ):
        env = [
            V1EnvVar(
                name="MY_POD_IP",
                value_from=V1EnvVarSource(
                    field_ref=V1ObjectFieldSelector(field_path="status.podIP")
                ),
            )
        ]
        pod = self._create_pod(
            "elasticdl-" + job_name + "-master",
            job_name,
            image_name,
            ["python"],
            parse_resource(master_resource_request),
            parse_resource(master_resource_limit),
            args,
            master_pod_priority,
            image_pull_policy,
            restart_policy,
            volume_name,
            mount_path,
            None,
            env,
        )
        resp = self._v1.create_namespaced_pod(self._ns, pod)
        print("Master launched. status='%s'" % str(resp.status))

    def create_worker(
        self,
        worker_id,
        resource_requests,
        resource_limits,
        priority=None,
        mount_path=None,
        volume_name=None,
        image_pull_policy=None,
        command=None,
        args=None,
        restart_policy="OnFailure",
    ):
        self._logger.info("Creating worker: " + str(worker_id))
        # Find that master pod that will be used as the owner reference
        # for this worker pod.
        pods = self._v1.list_namespaced_pod(
            namespace=self._ns,
            label_selector="elasticdl_job_name=" + self._job_name,
        ).items
        master_pod = [
            pod
            for pod in pods
            if (pod.metadata.name == self.get_master_pod_name())
        ]
        self._logger.info("image pull policy : " + image_pull_policy)
        pod = self._create_pod(
            self.get_worker_pod_name(worker_id),
            self._job_name,
            self._image,
            command,
            resource_requests,
            resource_limits,
            args,
            priority,
            image_pull_policy,
            restart_policy,
            volume_name,
            mount_path,
            master_pod,
            None,
        )
        return self._v1.create_namespaced_pod(self._ns, pod)

    def delete_worker(self, worker_id):
        self._logger.info("Deleting worker: " + str(worker_id))
        self._v1.delete_namespaced_pod(
            self.get_worker_pod_name(worker_id),
            self._ns,
            body=client.V1DeleteOptions(grace_period_seconds=0),
        )
