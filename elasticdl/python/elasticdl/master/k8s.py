import logging
import os
import threading

from kubernetes import client, config, watch


class Client(object):
    def __init__(
        self, *, worker_image, namespace, job_name, event_callback
    ):
        """
        ElasticDL k8s client.

        Args:
            worker_image: Docker image path for ElasticDL workers.
            namespace: k8s namespace for ElasticDL pods.
            job_name: ElasticDL job name, should be unique in the namespace.
                Used as worker pod name prefix and value for "elasticdl" label.
            master_addr: Master's ip:port.
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
        self._logger = logging.getLogger("k8s")
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
            self._event_cb(event)

    def get_pod_name(self, worker_id):
        return "elasticdl-worker-" + self._job_name + "-" + str(worker_id)

    def _create_worker_pod(self, worker_id, resource_requests, resource_limits, priority,
                           mount_path, volume_name, command, args, restart_policy):
        # Worker container config
        container = client.V1Container(
            name=self.get_pod_name(worker_id),
            image=self._image,
            command=command,
            resources=client.V1ResourceRequirements(
                requests=resource_requests,
                limits=resource_limits
            ),
            args=args
        )

        # Pod
        spec = client.V1PodSpec(
            containers=[container],
            restart_policy=restart_policy,
        )

        # Mount data path
        if mount_path is not None and volume_name is not None:
            volume = client.V1Volume(
                name='data-volume',
                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                    claim_name="fileserver-claim", read_only=False))
            spec.volumes = [volume]
            container.volume_mounts = [client.V1VolumeMount(name=volume_name, mount_path=mount_path)]

        if priority is not None:
            spec.priority_class_name = priority

        pod = client.V1Pod(
            spec=spec,
            metadata=client.V1ObjectMeta(
                name=self.get_pod_name(worker_id),
                labels={
                    "app": "elasticdl",
                    "elasticdl_job_name": self._job_name
                },
            ),
        )
        return pod

    def create_worker(self, worker_id, resource_requests, resource_limits, priority=None,
                      mount_path=None, volume_name=None, command=None, args=None,
                      restart_policy="OnFailure"):
        self._logger.info("Creating worker: " + str(worker_id))
        pod = self._create_worker_pod(
            worker_id, resource_requests, resource_limits, priority,
            mount_path, volume_name, command=command,
            args=args, restart_policy=restart_policy)
        self._v1.create_namespaced_pod(self._ns, pod)

    def delete_worker(self, worker_id):
        self._logger.info("Deleting worker: " + str(worker_id))
        self._v1.delete_namespaced_pod(
            self.get_pod_name(worker_id),
            self._ns,
            body=client.V1DeleteOptions(grace_period_seconds=0),
        )
