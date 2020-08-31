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

import threading
import time

from kubernetes import client, watch

from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl_client.common.k8s_client import (
    ELASTICDL_APP_NAME,
    ELASTICDL_JOB_KEY,
    ELASTICDL_REPLICA_INDEX_KEY,
    ELASTICDL_REPLICA_TYPE_KEY,
)
from elasticdl_client.common.k8s_client import Client as BaseClient
from elasticdl_client.common.k8s_client import append_pod_ip_to_env

_PS_SERVICE_PORT = 2222


def get_worker_pod_name(job_name, worker_id):
    return "elasticdl-%s-worker-%s" % (job_name, str(worker_id))


def get_ps_pod_name(job_name, ps_id):
    return "elasticdl-%s-ps-%s" % (job_name, str(ps_id))


class Client(BaseClient):
    def __init__(
        self,
        *,
        image_name,
        namespace,
        job_name,
        event_callback=None,
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
            event_callback: If not None, an event watcher will be created and
                events passed to the callback.
            force_use_kube_config_file: If true, force to load the cluster
                config from ~/.kube/config. Otherwise, if it's in a process
                running in a K8S environment, it loads the incluster config,
                if not, it loads the kube config file.
        """
        super().__init__(
            image_name=image_name,
            namespace=namespace,
            job_name=job_name,
            cluster_spec=cluster_spec,
            force_use_kube_config_file=force_use_kube_config_file,
        )
        self._event_cb = event_callback
        if self._event_cb:
            threading.Thread(
                target=self._watch, name="event_watcher", daemon=True
            ).start()

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
            except Exception as e:
                logger.debug(e)
            # In case of any flaky issue causing exceptions, we wait for little
            # time and retry.
            time.sleep(5)

    def _get_service_address(self, service_name, port):
        return "%s.%s.svc:%d" % (service_name, self.namespace, port)

    def get_worker_pod_name(self, worker_id):
        return get_worker_pod_name(self.job_name, worker_id)

    def get_ps_pod_name(self, ps_id):
        return get_ps_pod_name(self.job_name, ps_id)

    def get_ps_service_name(self, ps_id):
        return self.get_ps_pod_name(ps_id)

    def get_ps_service_address(self, ps_id):
        return self._get_service_address(
            self.get_ps_service_name(ps_id), _PS_SERVICE_PORT
        )

    def get_master_pod(self):
        return self.get_pod(self.get_master_pod_name())

    def get_worker_pod(self, worker_id):
        return self.get_pod(self.get_worker_pod_name(worker_id))

    def get_ps_pod(self, ps_id):
        return self.get_pod(self.get_ps_pod_name(ps_id))

    def get_pod(self, pod_name):
        try:
            return self.client.read_namespaced_pod(
                namespace=self.namespace, name=pod_name
            )
        except client.api_client.ApiException as e:
            logger.warning(
                "Exception when reading pod %s: %s\n" % (pod_name, e)
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

    def _create_ps_worker_pod(self, pod_name, type_key, index_key, **kargs):
        # Find that master pod that will be used as the owner reference
        # for the ps or worker pod.
        master_pod = self.get_master_pod()
        env = kargs["envs"] if "envs" in kargs else None
        env = append_pod_ip_to_env(env)
        pod = self.create_pod(
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
            termination_period=kargs.get("termination_period", None),
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

    def create_ps(self, **kargs):
        pod_name = self.get_ps_pod_name(kargs["ps_id"])
        return self._create_ps_worker_pod(
            pod_name, "ps", kargs["ps_id"], **kargs
        )

    def delete_worker(self, worker_id):
        self.delete_pod(self.get_worker_pod_name(worker_id))

    def delete_ps(self, ps_id):
        self.delete_pod(self.get_ps_pod_name(ps_id))

    def delete_pod(self, pod_name):
        self.client.delete_namespaced_pod(
            pod_name,
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
        selector = {
            "app": ELASTICDL_APP_NAME,
            ELASTICDL_JOB_KEY: self.job_name,
            ELASTICDL_REPLICA_TYPE_KEY: kargs["replica_type"],
        }
        if kargs["replica_index"] is not None:
            selector[ELASTICDL_REPLICA_INDEX_KEY] = str(kargs["replica_index"])
        spec = client.V1ServiceSpec(
            ports=[
                client.V1ServicePort(
                    port=kargs["port"], target_port=kargs["target_port"]
                )
            ],
            selector=selector,
            type=kargs.get("service_type", None),
        )
        service = client.V1Service(
            api_version="v1", kind="Service", metadata=metadata, spec=spec
        )
        if self.cluster:
            service = self.cluster.with_service(service)
        return self.client.create_namespaced_service(self.namespace, service)

    def get_master_log(self):
        return self.get_pod_log(self.get_master_pod_name())

    def get_worker_log(self, worker_id):
        return self.get_pod_log(self.get_worker_pod_name(worker_id))

    def get_ps_log(self, ps_id):
        return self.get_pod_log(self.get_ps_pod_name(ps_id))

    def get_pod_log(self, pod_name):
        try:
            return self.client.read_namespaced_pod_log(
                namespace=self.namespace, name=pod_name
            )
        except client.api_client.ApiException as e:
            logger.warning(
                "Exception when reading log of pod %s: %s\n" % (pod_name, e)
            )
            return None
