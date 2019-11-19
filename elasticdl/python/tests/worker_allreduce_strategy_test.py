import os
import unittest

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.args import parse_worker_args
from elasticdl.python.common.constants import DistributionStrategy
from elasticdl.python.tests.test_utils import get_mnist_dataset
from elasticdl.python.worker.worker import Worker


class WorkerAllReduceStrategyTest(unittest.TestCase):
    def setUp(self):
        self._model_zoo_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../../../model_zoo"
        )
        self._model_def = (
            "mnist_functional_api.mnist_functional_api.custom_model"
        )
        self._batch_size = 16
        self._test_steps = 10
        self._workers = []
        self._create_worker(2)

    def _create_worker(self, worker_num):
        for i in range(worker_num):
            arguments = [
                "--worker_id",
                i,
                "--job_type",
                elasticdl_pb2.TRAINING,
                "--minibatch_size",
                self._batch_size,
                "--model_zoo",
                self._model_zoo_path,
                "--model_def",
                self._model_def,
                "--distribution_strategy",
                DistributionStrategy.ALLREDUCE,
            ]
            args = parse_worker_args(arguments)
            worker = Worker(args)
            self._workers.append(worker)

    def test_collect_gradients_with_allreduce(self):
        worker = self._workers[0]
        train_db, _ = get_mnist_dataset(self._batch_size)
        for step, (x, y) in enumerate(train_db):
            if step == 0:
                worker._run_model_call_before_training(x)
            w_loss, w_grads = worker.training_process_eagerly(x, y)
            if step == self._test_steps:
                break
            self.assertEqual(
                worker._collect_gradients_with_allreduce(w_grads), (True, None)
            )


if __name__ == "__main__":
    unittest.main()
