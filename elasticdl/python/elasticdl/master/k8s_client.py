import logging
import os
import threading
import traceback

from kubernetes import client, config, watch
from kubernetes.client import (
    V1PersistentVolumeClaimVolumeSource as pvcVolumeSource,
)

WORKER_POD_NAME_PREFIX = "elasticdl-worker-"


class Client(object):
    def __init__(self, *, worker_image, namespace, job_name, event_callback):
        """
        ElasticDL k8s client.

        Args:
            worker_image: Docker image path for ElasticDL workers.
            namespace: k8s namespace for ElasticDL pods.
            job_name: ElasticDL job name, should be unique in the namespace.
                Used as worker pod name prefix and value for "elasticdl" label.
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
        self._image = worker_image
        self._ns = namespace
        self._job_name = job_name
        self._event_cb = event_callback
        if self._event_cb:
            threading.Thread(
                target=self._watch, name="event_watcher", daemon=True
            ).start()

    def _watch(self):
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

    def get_worker_pod_name(self, worker_id):
        return WORKER_POD_NAME_PREFIX + self._job_name + "-" + str(worker_id)

    def _create_worker_pod(
        self,
        worker_id,
        resource_requests,
        resource_limits,
        priority,
        mount_path,
        volume_name,
        image_pull_policy,
        command,
        args,
        restart_policy,
    ):
        # Worker container config
        container = client.V1Container(
            name=self.get_worker_pod_name(worker_id),
            image=self._image,
            command=command,
            resources=client.V1ResourceRequirements(
                requests=resource_requests, limits=resource_limits
            ),
            image_pull_policy=image_pull_policy,
            args=args,
        )

        # Pod
        spec = client.V1PodSpec(
            containers=[container], restart_policy=restart_policy
        )

        # Mount data path
        if mount_path is not None and volume_name is not None:
            volume = client.V1Volume(
                name="data-volume",
                persistent_volume_claim=pvcVolumeSource(
                    claim_name="fileserver-claim", read_only=False
                ),
            )
            spec.volumes = [volume]
            container.volume_mounts = [
                client.V1VolumeMount(name=volume_name, mount_path=mount_path)
            ]

        if priority is not None:
            spec.priority_class_name = priority

        # Find that master pod that will be used as the owner reference
        # for this worker pod.
        pods = self._v1.list_namespaced_pod(
            namespace=self._ns,
            label_selector="elasticdl_job_name=" + self._job_name,
        ).items
        master_pod = [
            pod
            for pod in pods
            if (pod.metadata.name == "elasticdl-master-" + self._job_name)
        ]
        owner_ref = (
            [
                client.V1OwnerReference(
                    api_version="v1",
                    block_owner_deletion=True,
                    kind="Pod",
                    name=master_pod[0].metadata.name,
                    uid=master_pod[0].metadata.uid,
                )
            ]
            if len(master_pod) != 0
            else None
        )

        pod = client.V1Pod(
            spec=spec,
            metadata=client.V1ObjectMeta(
                name=self.get_worker_pod_name(worker_id),
                labels={
                    "app": "elasticdl",
                    "elasticdl_job_name": self._job_name,
                },
                # TODO: Add tests for this once we've done refactoring on
                # k8s client code and the constant strings
                owner_references=owner_ref,
            ),
        )
        return pod

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
        pod = self._create_worker_pod(
            worker_id,
            resource_requests,
            resource_limits,
            priority,
            mount_path,
            volume_name,
            image_pull_policy,
            command=command,
            args=args,
            restart_policy=restart_policy,
        )
        return self._v1.create_namespaced_pod(self._ns, pod)

    def delete_worker(self, worker_id):
        self._logger.info("Deleting worker: " + str(worker_id))
        self._v1.delete_namespaced_pod(
            self.get_worker_pod_name(worker_id),
            self._ns,
            body=client.V1DeleteOptions(grace_period_seconds=0),
        )
