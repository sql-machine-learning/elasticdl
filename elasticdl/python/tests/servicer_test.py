import random
import unittest
from collections import defaultdict

import numpy as np
import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.tensor import tensor_pb_to_ndarray
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.master.task_dispatcher import _TaskDispatcher


def _get_variable_names(model_pb):
    return [v.name for v in model_pb.param]


class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__(name="test_model")
        self.dense_1 = tf.keras.layers.Dense(
            32, activation="relu", name="dense_1"
        )
        self.dense_2 = tf.keras.layers.Dense(
            1, activation="sigmoid", name="dense_2"
        )

    def call(self, inputs):
        x = self.dense_1(inputs)
        return self.dense_2(x)

    @staticmethod
    def input_shapes():
        return 10, 10

    @staticmethod
    def optimizer(lr=0.1):
        return tf.optimizers.SGD(lr)


class ServicerTest(unittest.TestCase):
    def testGetEmptyTask(self):
        master = MasterServicer(
            3,
            _TaskDispatcher({}, {}, {}, records_per_task=3, num_epochs=2),
            evaluation_service=None,
        )

        req = elasticdl_pb2.GetTaskRequest()

        # No task yet, make sure the returned versions are as expected.
        req.worker_id = 1
        task = master.GetTask(req, None)
        self.assertEqual("", task.shard_name)
        self.assertEqual(0, task.model_version)

        master._version = 1
        task = master.GetTask(req, None)
        self.assertEqual("", task.shard_name)
        self.assertEqual(1, task.model_version)

    def _check_get_model_response(self, version, expected, response):
        self.assertEqual(version, response.version)
        self.assertEqual(
            list(sorted(expected.keys())),
            sorted([v.name for v in response.param]),
        )
        for var in response.param:
            exp_value = expected[var.name]
            np.testing.assert_array_equal(exp_value, tensor_pb_to_ndarray(var))

    def testReportTaskResult(self):
        task_d = _TaskDispatcher(
            {"shard_1": (0, 10), "shard_2": (0, 9)},
            {},
            {},
            records_per_task=3,
            num_epochs=2,
        )
        master = MasterServicer(3, task_d, evaluation_service=None,)

        # task to number of runs.
        tasks = defaultdict(int)
        while True:
            req = elasticdl_pb2.GetTaskRequest()
            req.worker_id = random.randint(1, 10)
            task = master.GetTask(req, None)
            if not task.shard_name:
                break
            self.assertEqual(task_d._doing[task.task_id][0], req.worker_id)
            task_key = (task.shard_name, task.start, task.end)
            tasks[task_key] += 1
            report = elasticdl_pb2.ReportTaskResultRequest()
            report.task_id = task.task_id
            if task.start == 0 and tasks[task_key] == 1:
                # Simulate error reports.
                report.err_message = "Worker error"
            master.ReportTaskResult(report, None)

        self.assertDictEqual(
            {
                ("shard_1", 0, 3): 3,
                ("shard_1", 3, 6): 2,
                ("shard_1", 6, 9): 2,
                ("shard_1", 9, 10): 2,
                ("shard_2", 0, 3): 3,
                ("shard_2", 3, 6): 2,
                ("shard_2", 6, 9): 2,
            },
            tasks,
        )


if __name__ == "__main__":
    unittest.main()
