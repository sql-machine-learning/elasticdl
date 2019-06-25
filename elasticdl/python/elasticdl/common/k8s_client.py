import logging
import os
import threading
import traceback
import time

from kubernetes import client, config, watch
from kubernetes.client import (
    V1PersistentVolumeClaimVolumeSource as pvcVolumeSource,
    V1EnvVar,
    V1EnvVarSource,
    V1ObjectFieldSelector,
)
from elasticdl.python.elasticdl.common.k8s_utils import parse_resource

ELASTICDL_JOB_KEY = "elasticdl_job_name"


class Client(object):
    def __init__(self, *, image_name, namespace, job_name, event_callback):
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
        stream = watch.Watch().stream(
            self._v1.list_namespaced_pod,
            self._ns,
            label_selector=ELASTICDL_JOB_KEY + "=" + self._job_name,
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

    def _get_master_pod(self):
        try:
            return self._v1.read_namespaced_pod(
                name=self.get_master_pod_name(), namespace=self._ns
            )
        except client.api_client.ApiException as e:
            self._logger.warning("Exception when reading master pod: %s\n" % e)
            return None

    @staticmethod
    def _create_owner_reference(owner_pod):
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
        container = client.V1Container(
            name=kargs["pod_name"],
            image=kargs["image_name"],
            command=kargs["command"],
            resources=client.V1ResourceRequirements(
                requests=kargs["resource_requests"],
                limits=kargs["resource_limits"],
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
        if all([kargs["volume_name"], kargs["mount_path"]]):
            volume = client.V1Volume(
                name=kargs["volume_name"],
                persistent_volume_claim=pvcVolumeSource(
                    claim_name="fileserver-claim", read_only=False
                ),
            )
            spec.volumes = [volume]
            container.volume_mounts = [
                client.V1VolumeMount(
                    name=kargs["volume_name"], mount_path=kargs["mount_path"]
                )
            ]
        elif any([kargs["volume_name"], kargs["mount_path"]]):
            raise ValueError(
                "Not both of the parameters volume_name and "
                "mount_path are provided."
            )

        pod = client.V1Pod(
            spec=spec,
            metadata=client.V1ObjectMeta(
                name=kargs["pod_name"],
                labels={
                    "app": "elasticdl",
                    ELASTICDL_JOB_KEY: kargs["job_name"],
                },
                owner_references=self._create_owner_reference(
                    kargs["owner_pod"]
                ),
                namespace=self._ns,
            ),
        )
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
        pod = self._create_pod(
            pod_name="elasticdl-" + kargs["job_name"] + "-master",
            job_name=kargs["job_name"],
            image_name=kargs["image_name"],
            command=["python"],
            resource_requests=parse_resource(kargs["resource_requests"]),
            resource_limits=parse_resource(kargs["resource_limits"]),
            container_args=kargs["args"],
            pod_priority=kargs["pod_priority"],
            image_pull_policy=kargs["image_pull_policy"],
            restart_policy=kargs["restart_policy"],
            volume_name=kargs["volume_name"],
            mount_path=kargs["mount_path"],
            owner_pod=None,
            env=env,
        )
        resp = self._v1.create_namespaced_pod(self._ns, pod)
        self._logger.info("Master launched. status='%s'" % str(resp.status))

    def create_worker(self, **kargs):
        # Find that master pod that will be used as the owner reference
        # for this worker pod.
        master_pod = self._get_master_pod()
        pod = self._create_pod(
            pod_name=self.get_worker_pod_name(kargs["worker_id"]),
            job_name=self._job_name,
            image_name=self._image,
            command=kargs["command"],
            resource_requests=kargs["resource_requests"],
            resource_limits=kargs["resource_limits"],
            container_args=kargs["args"],
            pod_priority=kargs["priority"],
            image_pull_policy=kargs["image_pull_policy"],
            restart_policy=kargs["restart_policy"],
            volume_name=kargs["volume_name"],
            mount_path=kargs["mount_path"],
            owner_pod=master_pod,
            env=None,
        )
        return self._v1.create_namespaced_pod(self._ns, pod)

    def delete_worker(self, worker_id):
        self._v1.delete_namespaced_pod(
            self.get_worker_pod_name(worker_id),
            self._ns,
            body=client.V1DeleteOptions(grace_period_seconds=0),
        )

    def _get_tensorboard_service_name(self):
        return "tensorboard-" + self._job_name

    def create_tensorboard_service(self, port=80, target_port=6006):
        self._v1.create_namespaced_service(
            self._ns,
            client.V1Service(
                api_version="v1",
                kind="Service",
                metadata=client.V1ObjectMeta(
                    name=self._get_tensorboard_service_name(),
                    labels={
                        "app": "elasticdl",
                        ELASTICDL_JOB_KEY: self._job_name
                    },
                    owner_references=self._create_owner_reference(
                        self._get_master_pod()
                    ),
                    namespace=self._ns,
                ),
                spec=client.V1ServiceSpec(
                    ports=[
                        client.V1ServicePort(
                            port=port,
                            target_port=target_port
                        )
                    ],
                    selector={
                        ELASTICDL_JOB_KEY: self._job_name
                    },
                    type="LoadBalancer",
                ),
            ),
        )

    def get_tensorboard_external_ip(self):
        current_wait_secs = 0
        check_interval = 5
        max_wait_secs = 120
        service = self._v1.read_namespaced_service(
            name=self._get_tensorboard_service_name(),
            namespace=self._ns
        ).to_dict()
        self._logger.info(
            "Waiting for pending external IP of TensorBoard service..."
        )
        while service["status"]["load_balancer"]["ingress"] is None:
            service = self._v1.read_namespaced_service(
                name=self._get_tensorboard_service_name(),
                namespace=self._ns
            ).to_dict()
            time.sleep(check_interval)
            current_wait_secs += check_interval
            if current_wait_secs > max_wait_secs:
                raise Exception(
                    "Unable to get an external IP for TensorBoard service"
                )
        return service["status"]["load_balancer"]["ingress"][0]["ip"]
