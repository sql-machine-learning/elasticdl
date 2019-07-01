import random
import unittest
import numpy as np
import os
import tempfile

from collections import defaultdict

import tensorflow as tf

from elasticdl.python.master.task_queue import _TaskQueue
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.master.checkpoint_service import CheckpointService
from elasticdl.python.common.ndarray import ndarray_to_tensor
from elasticdl.python.common.ndarray import tensor_to_ndarray
from elasticdl.proto import elasticdl_pb2


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
            2,
            3,
            None,
            _TaskQueue({}, {}, {}, records_per_task=3, num_epochs=2),
            init_var=[],
            checkpoint_filename_for_init="",
            checkpoint_service=CheckpointService("", 0, 0, False),
            evaluation_service=None,
        )

        req = elasticdl_pb2.GetTaskRequest()

        # No task yet, make sure the returned versions are as expected.
        req.worker_id = 1
        task = master.GetTask(req, None)
        self.assertEqual("", task.shard_file_name)
        self.assertEqual(0, task.model_version)

        master._version = 1
        task = master.GetTask(req, None)
        self.assertEqual("", task.shard_file_name)
        self.assertEqual(1, task.model_version)

    def testGetModel(self):
        master = MasterServicer(
            2,
            3,
            None,
            None,
            init_var=[],
            checkpoint_filename_for_init="",
            checkpoint_service=CheckpointService("", 0, 0, False),
            evaluation_service=None,
        )
        master.set_model_var("x", np.array([1.0, 1.0], dtype=np.float32))
        # Now master model is version 0
        self.assertEqual(0, master._version)

        # Get version 0 with minimum method
        req = elasticdl_pb2.GetModelRequest()
        req.version = 0
        req.method = elasticdl_pb2.MINIMUM
        model = master.GetModel(req, None)
        self.assertEqual(0, model.version)
        self.assertEqual(["x"], list(model.param.keys()))
        np.testing.assert_array_equal(
            np.array([1.0, 1.0]), tensor_to_ndarray(model.param["x"])
        )

        # Increase master model version to 1, but still request
        # version 0 with minimum method, we should get version 1
        master._version = 1
        master.set_model_var("x", np.array([2.0, 2.0], dtype=np.float32))
        master.set_model_var("y", np.array([12.0, 13.0], dtype=np.float32))
        model = master.GetModel(req, None)
        self.assertEqual(1, model.version)
        self.assertEqual(["x", "y"], list(sorted(model.param.keys())))
        np.testing.assert_array_equal(
            np.array([2.0, 2.0]), tensor_to_ndarray(model.param["x"])
        )
        np.testing.assert_array_equal(
            np.array([12.0, 13.0]), tensor_to_ndarray(model.param["y"])
        )

        # Try to get version 2, it should raise exception.
        req.version = 2
        self.assertRaises(ValueError, master.GetModel, req, None)

        # Get fixed version 1
        req.method = elasticdl_pb2.FIXED
        req.version = 1
        model = master.GetModel(req, None)
        self.assertEqual(1, model.version)
        self.assertEqual(["x", "y"], list(sorted(model.param.keys())))
        np.testing.assert_array_equal(
            np.array([2.0, 2.0]), tensor_to_ndarray(model.param["x"])
        )
        np.testing.assert_array_equal(
            np.array([12.0, 13.0]), tensor_to_ndarray(model.param["y"])
        )

        # Previous model unavailable due to no checkpoint
        req.version = 0
        model = master.GetModel(req, None)
        self.assertFalse(model.param)

        # Previous model available through checkpoint
        with tempfile.TemporaryDirectory() as tempdir:
            chk_dir = os.path.join(tempdir, "testGetModel")
            os.makedirs(chk_dir)
            req.version = master._version
            req.method = elasticdl_pb2.MINIMUM
            model = master.GetModel(req, None)
            master._checkpoint_service = CheckpointService(
                chk_dir, 2, 5, False
            )
            master._checkpoint_service.save(master._version, model, False)
            master._version = 2
            master.set_model_var("z", np.array([2.0, 2.0], dtype=np.float32))
            req.version = 1
            req.method = elasticdl_pb2.FIXED
            model = master.GetModel(req, None)
            self.assertEqual(1, model.version)
            self.assertEqual(["x", "y"], list(sorted(model.param.keys())))
            np.testing.assert_array_equal(
                np.array([2.0, 2.0]), tensor_to_ndarray(model.param["x"])
            )
            np.testing.assert_array_equal(
                np.array([12.0, 13.0]), tensor_to_ndarray(model.param["y"])
            )

    def testReportGradient(self):
        def makeGrad():
            """ Make a ReportGradientRequest compatible with model"""
            req = elasticdl_pb2.ReportGradientRequest()
            req.gradient["x"].CopyFrom(
                ndarray_to_tensor(np.array([0.1], dtype=np.float32))
            )
            req.gradient["y"].CopyFrom(
                ndarray_to_tensor(np.array([0.03, 0.06], dtype=np.float32))
            )
            req.model_version = 1
            return req

        master = MasterServicer(
            3,
            3,
            tf.optimizers.SGD(0.1),
            None,
            init_var=[],
            checkpoint_filename_for_init="",
            checkpoint_service=CheckpointService("", 0, 0, False),
            evaluation_service=None,
        )
        master._version = 1
        master.set_model_var("x", np.array([2.0], dtype=np.float32))
        master.set_model_var("y", np.array([12.0, 13.0], dtype=np.float32))

        # Report a future version, should raise exception
        req = makeGrad()
        req.model_version = 2
        self.assertRaises(ValueError, master.ReportGradient, req, None)

        # Report an old version, should not be accepted
        req = makeGrad()
        req.model_version = 0
        res = master.ReportGradient(req, None)
        self.assertFalse(res.accepted)
        self.assertEqual(1, res.model_version)

        # Report a unknown gradient, should raise.
        req = makeGrad()
        req.gradient["z"].CopyFrom(
            ndarray_to_tensor(np.array([0.1], dtype=np.float32))
        )
        self.assertRaises(ValueError, master.ReportGradient, req, None)

        # Report an incompatible gradient, should raise.
        req = makeGrad()
        req.gradient["y"].CopyFrom(
            ndarray_to_tensor(np.array([0.1], dtype=np.float32))
        )
        self.assertRaises(ValueError, master.ReportGradient, req, None)

        # Report a current version, should be accepted.
        req = makeGrad()
        res = master.ReportGradient(req, None)
        self.assertTrue(res.accepted)
        self.assertEqual(1, res.model_version)

        # Report a current version with part of gradients, should be accepted.
        req = makeGrad()
        del req.gradient["y"]
        res = master.ReportGradient(req, None)
        self.assertTrue(res.accepted)
        self.assertEqual(1, res.model_version)
        # Gradient should be accumulated.
        np.testing.assert_array_equal(
            np.array([0.2], dtype=np.float32), master._gradient_sum["x"]
        )
        np.testing.assert_array_equal(
            np.array([0.03, 0.06], dtype=np.float32), master._gradient_sum["y"]
        )
        self.assertEqual(2, master._grad_n)

        # Report a current version, should be accepted, and a new version
        # created
        req = makeGrad()
        res = master.ReportGradient(req, None)
        self.assertTrue(res.accepted)
        self.assertEqual(2, res.model_version)
        self.assertFalse(master._gradient_sum)
        self.assertEqual(0, master._grad_n)
        np.testing.assert_array_equal(
            # [2] - 0.1 * [0.1]
            np.array([1.99], dtype=np.float32),
            master._model["x"].numpy(),
        )
        np.testing.assert_array_equal(
            # [12, 13] - 0.1 * [0.02, 0.04]
            np.array([11.998, 12.996], dtype=np.float32),
            master._model["y"].numpy(),
        )

    def testReportTaskResult(self):
        task_q = _TaskQueue(
            {"shard_1": 10, "shard_2": 9},
            {},
            {},
            records_per_task=3,
            num_epochs=2,
        )
        master = MasterServicer(
            3,
            3,
            None,
            task_q,
            init_var=[],
            checkpoint_filename_for_init="",
            checkpoint_service=CheckpointService("", 0, 0, False),
            evaluation_service=None,
        )

        # task to number of runs.
        tasks = defaultdict(int)
        while True:
            req = elasticdl_pb2.GetTaskRequest()
            req.worker_id = random.randint(1, 10)
            task = master.GetTask(req, None)
            if not task.shard_file_name:
                break
            self.assertEqual(task_q._doing[task.task_id][0], req.worker_id)
            task_key = (task.shard_file_name, task.start, task.end)
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

    def testUserDefinedModel(self):
        master = MasterServicer(
            2,
            3,
            None,
            None,
            init_var=[],
            checkpoint_filename_for_init="",
            checkpoint_service=CheckpointService("", 0, 0, False),
            evaluation_service=None,
        )
        req = elasticdl_pb2.GetModelRequest()
        req.method = elasticdl_pb2.MINIMUM
        req.version = 0

        model_inst = SimpleModel()
        model_inst.build(SimpleModel.input_shapes())
        for variable in model_inst.trainable_variables:
            master.set_model_var(variable.name, variable.numpy())
        # Get version 0
        model = master.GetModel(req, None)
        self.assertEqual(0, model.version)
        self.assertEqual(
            [
                "dense_1/bias:0",
                "dense_1/kernel:0",
                "dense_2/bias:0",
                "dense_2/kernel:0",
            ],
            list(sorted(model.param.keys())),
        )


if __name__ == "__main__":
    unittest.main()
