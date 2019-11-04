import os
import random
import time
import unittest

from elasticdl.python.common.k8s_tensorboard_client import TensorBoardClient


@unittest.skipIf(
    os.environ.get("K8S_TESTS", "True") == "False",
    "No Kubernetes cluster available",
)
class K8sTensorBoardClientTest(unittest.TestCase):
    def test_create_tensorboard_service(self):
        tb_client = TensorBoardClient(
            image_name=None,
            namespace="default",
            job_name="test-job-%d-%d"
            % (int(time.time()), random.randint(1, 101)),
            event_callback=None,
        )
        tb_client._k8s_client.create_tensorboard_service(
            port=80, service_type="LoadBalancer"
        )
        time.sleep(1)
        service = tb_client._get_tensorboard_service()
        self.assertTrue("load_balancer" in service["status"])
        self.assertEqual(service["spec"]["ports"][0]["port"], 80)


if __name__ == "__main__":
    unittest.main()
