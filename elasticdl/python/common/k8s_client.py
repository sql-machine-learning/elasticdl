import os
import threading
import time
import traceback

import yaml
from kubernetes import client, config, watch
from kubernetes.client import V1EnvVar, V1EnvVarSource, V1ObjectFieldSelector

from elasticdl.python.common.k8s_resource import parse as parse_resource
from elasticdl.python.common.k8s_volume import parse_volume_and_mount
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.common.model_utils import load_module

ELASTICDL_APP_NAME = "elasticdl"
ELASTICDL_JOB_KEY = "elasticdl-job-name"
ELASTICDL_REPLICA_TYPE_KEY = "elasticdl-replica-type"
ELASTICDL_REPLICA_INDEX_KEY = "elasticdl-replica-index"
_PS_SERVICE_PORT = 2222
_WORKER_SERVICE_PORT = 3333


def get_master_pod_name(job_name):
    return "elasticdl-%s-master" % job_name


def get_worker_pod_name(job_name, worker_id):
    return "elasticdl-%s-worker-%s" % (job_name, str(worker_id))


def get_ps_pod_name(job_name, ps_id):
    return "elasticdl-%s-ps-%s" % (job_name, str(ps_id))


def get_env_with_ip(**kargs):
    env = [
        V1EnvVar(
            name="MY_POD_IP",
            value_from=V1EnvVarSource(
                field_ref=V1ObjectFieldSelector(field_path="status.podIP")
            ),
        )
    ]
    if "envs" in kargs:
        for key in kargs["envs"]:
            env.append(V1EnvVar(name=key, value=kargs["envs"][key]))
    return env


class Client(object):
    def __init__(
        self,
        *,
        image_name,
        namespace,
        job_name,
        event_callback=None,
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
        try:
            if os.getenv("KUBERNETES_SERVICE_HOST"):
                # We are running inside a k8s cluster
                config.load_incluster_config()
            else:
                # Use user's kube config
                config.load_kube_config()
        except Exception as ex:
            traceback.print_exc()
            raise Exception(
                "Failed to load configuration for Kubernetes:\n%s" % str(ex)
            )

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
        while True:
            try:
                stream = watch.Watch().stream(
                    self.client.list_namespaced_pod,
                    self.namespace,
                    label_selector=ELASTICDL_JOB_KEY + "=" + self.job_name,
                )
                for event in stream:
                    self._event_cb(event)
            except Exception:
                traceback.print_exc()
            # In case of any flaky issue causing exceptions, we wait for little
            # time and retry.
            time.sleep(5)

    def _get_service_address(self, service_name, port):
        return "%s.%s.svc:%d" % (service_name, self.namespace, port)

    def get_master_pod_name(self):
        return get_master_pod_name(self.job_name)

    def get_worker_pod_name(self, worker_id):
        return get_worker_pod_name(self.job_name, worker_id)

    def get_worker_service_name(self, worker_id):
        return self.get_worker_pod_name(worker_id)

    def get_worker_service_address(self, worker_id):
        return self._get_service_address(
            self.get_worker_service_name(worker_id), _WORKER_SERVICE_PORT
        )

    def get_ps_pod_name(self, ps_id):
        return get_ps_pod_name(self.job_name, ps_id)

    def get_ps_service_name(self, ps_id):
        return self.get_ps_pod_name(ps_id)

    def get_ps_service_address(self, ps_id):
        return self._get_service_address(
            self.get_ps_service_name(ps_id), _PS_SERVICE_PORT
        )

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

    def get_ps_pod(self, ps_id):
        try:
            return self.client.read_namespaced_pod(
                name=self.get_ps_pod_name(ps_id), namespace=self.namespace
            )
        except client.api_client.ApiException as e:
            logger.warning("Exception when reading ps pod: %s\n" % e)
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

    def get_ps_service(self, ps_id):
        try:
            return self.client.read_namespaced_service(
                # PS service has the same name as pod name
                name=self.get_ps_service_name(ps_id),
                namespace=self.namespace,
            )
        except client.api_client.ApiException as e:
            logger.warning("Exception when reading PS service: %s\n" % e)
            return None

    def get_worker_service(self, worker_id):
        try:
            return self.client.read_namespaced_service(
                # Worker service has the same name as pod name
                name=self.get_worker_service_name(worker_id),
                namespace=self.namespace,
            )
        except client.api_client.ApiException as e:
            logger.warning("Exception when reading worker service: %s\n" % e)
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
            yaml.safe_dump(pod_dict, f)

    def _create_master_pod_obj(self, **kargs):
        env = get_env_with_ip(kargs)

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
        pod.api_version = "v1"
        pod.kind = "Pod"
        return pod

    def _create_ps_worker_pod(self, pod_name, type_key, index_key, **kargs):
        # Find that master pod that will be used as the owner reference
        # for the ps or worker pod.
        master_pod = self.get_master_pod()
        env = get_env_with_ip(kargs)
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
            ps_addrs=kargs.get("ps_addrs", ""),
            env=env,
        )
        # Add replica type and index
        pod.metadata.labels[ELASTICDL_REPLICA_TYPE_KEY] = type_key
        pod.metadata.labels[ELASTICDL_REPLICA_INDEX_KEY] = str(index_key)
        return self.client.create_namespaced_pod(self.namespace, pod)

    def create_worker(self, **kargs):
        pod_name = self.get_worker_pod_name(kargs["worker_id"])
        return self._create_ps_worker_pod(
            pod_name, "worker", kargs["worker_id"], **kargs
        )

    def create_embedding_service(self, **kargs):
        pod_name = self.get_embedding_service_pod_name(kargs["worker_id"])
        return self._create_ps_worker_pod(
            pod_name, "embedding_service", kargs["worker_id"], **kargs
        )

    def create_ps(self, **kargs):
        pod_name = self.get_ps_pod_name(kargs["ps_id"])
        return self._create_ps_worker_pod(
            pod_name, "ps", kargs["ps_id"], **kargs
        )

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

    def delete_ps(self, ps_id):
        self.client.delete_namespaced_pod(
            self.get_ps_pod_name(ps_id),
            self.namespace,
            body=client.V1DeleteOptions(grace_period_seconds=0),
        )

    def get_tensorboard_service_name(self):
        return "tensorboard-" + self.job_name

    def create_tensorboard_service(
        self,
        port=80,
        target_port=6006,
        replica_type="master",
        replica_index="0",
        service_type="LoadBalancer",
    ):
        return self._create_service(
            name=self.get_tensorboard_service_name(),
            port=port,
            target_port=target_port,
            replica_type=replica_type,
            replica_index=replica_index,
            service_type=service_type,
            owner=self.get_master_pod(),
        )

    def create_ps_service(self, ps_id):
        return self._create_service(
            name=self.get_ps_service_name(ps_id),
            port=_PS_SERVICE_PORT,
            target_port=_PS_SERVICE_PORT,
            replica_type="ps",
            replica_index=ps_id,
            owner=self.get_ps_pod(ps_id),
        )

    def create_worker_service(self, worker_id):
        return self._create_service(
            name=self.get_worker_service_name(worker_id),
            port=_WORKER_SERVICE_PORT,
            target_port=_WORKER_SERVICE_PORT,
            replica_type="worker",
            replica_index=worker_id,
            owner=self.get_worker_pod(worker_id),
        )

    def _create_service(self, **kargs):
        labels = self._get_common_labels()

        metadata = client.V1ObjectMeta(
            name=kargs["name"],
            labels=labels,
            # Note: We have to add at least one annotation here.
            # Otherwise annotation is `None` and cannot be modified
            # using `with_service()` for cluster specific information.
            annotations=labels,
            owner_references=self.create_owner_reference(kargs["owner"])
            if "owner" in kargs
            else None,
            namespace=self.namespace,
        )
        spec = client.V1ServiceSpec(
            ports=[
                client.V1ServicePort(
                    port=kargs["port"], target_port=kargs["target_port"]
                )
            ],
            selector={
                "app": ELASTICDL_APP_NAME,
                ELASTICDL_JOB_KEY: self.job_name,
                ELASTICDL_REPLICA_TYPE_KEY: kargs["replica_type"],
                ELASTICDL_REPLICA_INDEX_KEY: str(kargs["replica_index"]),
            },
            type=kargs.get("service_type", None),
        )
        service = client.V1Service(
            api_version="v1", kind="Service", metadata=metadata, spec=spec
        )
        if self.cluster:
            service = self.cluster.with_service(service)
        return self.client.create_namespaced_service(self.namespace, service)

    def _get_common_labels(self):
        """Labels that should be attached to all k8s objects belong to
           current job.
        """
        return {"app": ELASTICDL_APP_NAME, ELASTICDL_JOB_KEY: self.job_name}
