import logging
import time

from kubernetes import client

from elasticdl.python.common import k8s_client as k8s


class TensorBoardClient(object):
    def __init__(self, k8s_client):
        """
        ElasticDL k8s TensorBoard client.

        Args:
            k8s_client: A Client object from elasticdl.python.common.k8s_client
        """
        self._k8s_client = k8s_client
        self._logger = logging.getLogger(__name__)

    def _get_tensorboard_service_name(self):
        return "tensorboard-" + self._k8s_client.job_name

    def create_tensorboard_service(
        self, port=80, target_port=6006, service_type="LoadBalancer"
    ):
        self._k8s_client.client.create_namespaced_service(
            self._k8s_client.namespace,
            client.V1Service(
                api_version="v1",
                kind="Service",
                metadata=client.V1ObjectMeta(
                    name=self._get_tensorboard_service_name(),
                    labels={
                        "app": k8s.ELASTICDL_APP_NAME,
                        k8s.ELASTICDL_JOB_KEY: self._k8s_client.job_name,
                    },
                    owner_references=k8s.Client.create_owner_reference(
                        self._k8s_client.get_master_pod()
                    ),
                    namespace=self._k8s_client.namespace,
                ),
                spec=client.V1ServiceSpec(
                    ports=[
                        client.V1ServicePort(
                            port=port, target_port=target_port
                        )
                    ],
                    selector={
                        k8s.ELASTICDL_JOB_KEY: self._k8s_client.job_name
                    },
                    type=service_type,
                ),
            ),
        )

    def _get_tensorboard_service(self):
        try:
            return self._k8s_client.client.read_namespaced_service(
                name=self._get_tensorboard_service_name(),
                namespace=self._k8s_client.namespace,
            ).to_dict()
        except client.api_client.ApiException as e:
            self._logger.warning(
                "Exception when reading TensorBoard service: %s\n" % e
            )
            return None

    def get_tensorboard_url(self, check_interval=5, wait_timeout=120):
        start_time = time.time()
        while True:
            if time.time() - start_time > wait_timeout:
                return None
            service = self._get_tensorboard_service()
            if (
                service is None
                or service["status"]["load_balancer"]["ingress"] is None
            ):
                time.sleep(check_interval)
            else:
                return service["status"]["load_balancer"]["ingress"][0]["ip"]
