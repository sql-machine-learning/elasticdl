import tempfile
import os
import unittest
import numpy as np
import recordio
import tensorflow as tf

from contextlib import closing
from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.model_helper import load_module
from elasticdl.python.master.task_queue import _TaskQueue
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.worker.worker import Worker
from elasticdl.python.tests.in_process_master import InProcessMaster
from elasticdl.python.master.checkpoint_service import (
    CheckpointService,
)

_module_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_module.py"
)
m = load_module(_module_file)


def create_recordio_file(size):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with closing(recordio.Writer(temp_file.name)) as f:
        for _ in range(size):
            x = np.random.rand(1).astype(np.float32)
            y = 2 * x + 1
            example_dict = {
                "x": tf.train.Feature(float_list=tf.train.FloatList(value=x)),
                "y": tf.train.Feature(float_list=tf.train.FloatList(value=y)),
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=example_dict)
            )
            f.write(example.SerializeToString())
    return temp_file.name


class WorkerTest(unittest.TestCase):
    def distributed_train_and_evaluate(self, training=True):
        """
        Run distributed training and evaluation with a local master.
        grpc calls are mocked by local master call.
        """

        class _Master(InProcessMaster):
            def ReportGradient(self, req):
                if 2 < self._m._version < 80:
                    # For testing of retrain when gradient not accepted.
                    # Increase master version to reject the gradient.
                    self._m._version += 1
                return self._m.ReportGradient(req, None)

            def ReportEvaluationMetrics(self, req):
                if 2 < self._m._version < 80:
                    # Testing of evaluation retries. Increase the master
                    # version so the evaluation metrics will not be accepted.
                    self._m._version += 1
                return self._m.ReportEvaluationMetrics(req, None)

        worker = Worker(1, _module_file, None)

        shards = {create_recordio_file(128): 128}
        if training:
            training_shards = shards
            evaluation_shards = {}
        else:
            training_shards = {}
            evaluation_shards = shards
        task_q = _TaskQueue(
            training_shards,
            evaluation_shards,
            records_per_task=64,
            num_epochs=1,
        )
        master = MasterServicer(
            2,
            16,
            worker._opt_fn(),
            task_q,
            init_var=[],
            checkpoint_filename_for_init="",
            checkpoint_service=CheckpointService("", 0, 0),
            evaluation_service=None,
        )
        worker._stub = _Master(master)

        for var in worker._model.trainable_variables:
            master.set_model_var(var.name, var.numpy())

        worker.run()

        req = elasticdl_pb2.GetTaskRequest()
        req.worker_id = 1
        task = master.GetTask(req, None)
        # No more task.
        self.assertTrue(not task.shard_file_name)

    def test_distributed_train_tf_example(self):
        self.distributed_train_and_evaluate(training=True)

    def test_distributed_evaluate_tf_example(self):
        self.distributed_train_and_evaluate(training=False)


if __name__ == "__main__":
    unittest.main()
