import logging
import os
import threading

from kubernetes import client, config, watch


class Client(object):
    def __init__(
        self, *, worker_image, namespace, job_name, master_addr, event_callback
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
        self._master_addr = master_addr
        self._event_cb = event_callback
        if self._event_cb:
            threading.Thread(
                target=self._watch, name="event_watcher", daemon=True
            ).start()

    def _watch(self):
        logger = logging.getLogger("k8s_event")
        stream = watch.Watch().stream(
            self._v1.list_namespaced_pod,
            self._ns,
            label_selector="elasticdl=" + self._job_name,
        )
        for event in stream:
            self._event_cb(event)

    def _get_pod_name(self, worker_name):
        return self._job_name + "-" + worker_name

    def _create_worker_pod(self, worker_name, command=None, args=None, restart_policy="OnFailure"):
        # Worker container config
        container = client.V1Container(
            name=self._get_pod_name(worker_name),
            image=self._image,
            command=command,
            args=args
        )
        # Pod
        pod = client.V1Pod(
            spec=client.V1PodSpec(
                containers=[container],
                restart_policy=restart_policy
            ),
            metadata=client.V1ObjectMeta(
                name=self._get_pod_name(worker_name),
                labels={"elasticdl": self._job_name},
            ),
        )
        return pod

    def create_worker(self, worker_name, command=None, args=None, restart_policy="OnFailure"):
        self._logger.warning("Creating worker: " + worker_name)
        pod = self._create_worker_pod(
            worker_name, command=command, args=args, restart_policy=restart_policy)
        self._v1.create_namespaced_pod(self._ns, pod)

    def delete_worker(self, worker_name):
        self._logger.warning("Deleting worker: " + worker_name)
        self._v1.delete_namespaced_pod(
            self._get_pod_name(worker_name),
            self._ns,
            body=client.V1DeleteOptions(grace_period_seconds=0),
        )
