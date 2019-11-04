import time

from kubernetes import client

from elasticdl.python.common import k8s_client as k8s
from elasticdl.python.common.log_utils import default_logger as logger


class TensorBoardClient(object):
    def __init__(self, **kwargs):
        """
        ElasticDL k8s TensorBoard client.

        Args:
            **kwargs: Additional keyword arguments passed to the
                `elasticdl.python.common.k8s_client.Client` object.
        """
        self._k8s_client = k8s.Client(**kwargs)

    def start_tensorboard_service(self):
        self._k8s_client.create_tensorboard_service()
        logger.info("Waiting for the URL for TensorBoard service...")
        tb_url = self._get_tensorboard_url()
        if tb_url:
            logger.info("TensorBoard service is available at: %s" % tb_url)
        else:
            logger.warning("Unable to get the URL for TensorBoard service")

    def _get_tensorboard_service(self):
        try:
            return self._k8s_client.client.read_namespaced_service(
                name=self._k8s_client.get_tensorboard_service_name(),
                namespace=self._k8s_client.namespace,
            ).to_dict()
        except client.api_client.ApiException as e:
            logger.warning(
                "Exception when reading TensorBoard service: %s\n" % e
            )
            return None

    def _get_tensorboard_url(self, check_interval=5, wait_timeout=120):
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
