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
from elasticdl_client.common.k8s_client import PodType, append_pod_ip_to_env

_PS_SERVICE_PORT = 2222
_WORKER_SERVICE_PORT = 3333
_EVALUATOR_SERVICE_PORT = 3333
_CHIEF_SERVICE_PORT = 3333


def get_pod_name(job_name, pod_type, worker_id):
    return "%s-edljob-%s-%s" % (job_name, pod_type, str(worker_id))


class Client(BaseClient):
    def __init__(
        self,
        *,
        image_name,
        namespace,
        job_name,
        event_callback=None,
        periodic_call_func=None,
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
            event_callback: If not None, an event watcher will be created and
                events passed to the callback.
            periodic_call_func: If not None, call this method periodically.
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
            cluster_spec_json=cluster_spec_json,
            force_use_kube_config_file=force_use_kube_config_file,
        )
        self._event_cb = event_callback
        self._periodic_call_func = periodic_call_func
        self._service_ports = {
            PodType.WORKER: _WORKER_SERVICE_PORT,
            PodType.EVALUATOR: _EVALUATOR_SERVICE_PORT,
            PodType.CHIEF: _CHIEF_SERVICE_PORT,
            PodType.PS: _PS_SERVICE_PORT,
        }

    def start_watch_events(self):
        if self._event_cb:
            threading.Thread(
                target=self._watch, name="event_watcher", daemon=True
            ).start()
        if self._periodic_call_func:
            threading.Thread(
                target=self._periodic_call, name="periodic_call", daemon=True
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

    def _periodic_call(self):
        while True:
            self._periodic_call_func()
            time.sleep(15)

    def get_pod_name(self, pod_type, id):
        return get_pod_name(self.job_name, pod_type, id)

    def get_service_name(self, pod_type, id):
        return self.get_pod_name(pod_type, id)

    def get_service_address(self, pod_type, id):
        service_name = self.get_service_name(pod_type, id)
        return "%s:%d" % (
            service_name,
            self._service_ports[pod_type],
        )

    def get_master_pod(self):
        return self.get_pod(self.get_master_pod_name())

    def get_typed_pod(self, pod_type, id):
        return self.get_pod(self.get_pod_name(pod_type, id))

    def get_pod(self, pod_name):
        try:
            return self.client.read_namespaced_pod(
                namespace=self.namespace, name=pod_name
            )
        except client.ApiException as e:
            logger.warning(
                "Exception when reading pod %s: %s\n" % (pod_name, e)
            )
            return None

    def get_ps_service(self, ps_id):
        try:
            return self.client.read_namespaced_service(
                # PS service has the same name as pod name
                name=self.get_service_name(PodType.PS, ps_id),
                namespace=self.namespace,
            )
        except client.ApiException as e:
            logger.warning("Exception when reading PS service: %s\n" % e)
            return None

    def get_worker_service(self, worker_id):
        try:
            return self.client.read_namespaced_service(
                # worker service has the same name as pod name
                name=self.get_service_name(PodType.WORKER, worker_id),
                namespace=self.namespace,
            )
        except client.ApiException as e:
            logger.warning("Exception when reading worker service: %s\n" % e)
            return None

    def create_typed_pod(self, pod_type, pod_id, **kargs):
        # Find that master pod that will be used as the owner reference
        # for the ps or worker pod.
        master_pod = self.get_master_pod()
        env = kargs["envs"] if "envs" in kargs else None
        env = append_pod_ip_to_env(env)
        pod_name = self.get_pod_name(pod_type, pod_id)

        pod = self.create_pod(
            pod_name=pod_name,
            job_name=self.job_name,
            image_name=kargs["image_name"],
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
            pod_type=pod_type,
        )
        # Add replica type and index
        pod.metadata.labels[ELASTICDL_REPLICA_TYPE_KEY] = pod_type
        pod.metadata.labels[ELASTICDL_REPLICA_INDEX_KEY] = str(pod_id)
        try:
            return self.client.create_namespaced_pod(self.namespace, pod)
        except client.rest.ApiException as e:
            logger.warning("Failed to create %s pod: %s\n" % (pod_name, e))
            return None

    def delete_typed_pod(self, pod_type, id):
        self.delete_pod(self.get_pod_name(pod_type, id))

    def delete_pod(self, pod_name):
        try:
            self.client.delete_namespaced_pod(
                pod_name,
                self.namespace,
                body=client.V1DeleteOptions(grace_period_seconds=0),
            )
        except Exception as e:
            logger.warning(e)

    def create_service(self, pod_type, id):
        # Use master pod as service owner so the service will not be
        #  deleted if the corresponding pod is deleted.
        service = self._create_service_obj(
            name=self.get_service_name(pod_type, id),
            port=self._service_ports[pod_type],
            target_port=self._service_ports[pod_type],
            replica_type=pod_type,
            replica_index=id,
            owner=self.get_master_pod(),
        )
        return self.client.create_namespaced_service(self.namespace, service)

    def patch_service(self, pod_type, original_id, id):
        service_name = self.get_service_name(pod_type, original_id)
        service = self._create_service_obj(
            name=service_name,
            port=self._service_ports[pod_type],
            target_port=self._service_ports[pod_type],
            replica_type=pod_type,
            replica_index=id,
            owner=self.get_master_pod(),
        )
        return self.client.patch_namespaced_service(
            service_name, self.namespace, service
        )

    def _create_service_obj(self, **kargs):
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
        service = self.cluster_spec.patch_service(service)
        return service

    def get_master_log(self):
        return self.get_pod_log(self.get_master_pod_name())

    def get_worker_log(self, worker_id):
        return self.get_pod_log(self.get_pod_name(PodType.WORKER, worker_id))

    def get_ps_log(self, ps_id):
        return self.get_pod_log(self.get_pod_name(PodType.PS, ps_id))

    def get_pod_log(self, pod_name):
        try:
            return self.client.read_namespaced_pod_log(
                namespace=self.namespace, name=pod_name
            )
        except client.ApiException as e:
            logger.warning(
                "Exception when reading log of pod %s: %s\n" % (pod_name, e)
            )
            return None
