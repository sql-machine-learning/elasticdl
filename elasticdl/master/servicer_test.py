import logging
import unittest
import numpy as np
import tensorflow as tf

from collections import defaultdict

import tensorflow as tf

tf.enable_eager_execution()

from google.protobuf import empty_pb2

from elasticdl.proto import master_pb2
from elasticdl.common.ndarray import ndarray_to_tensor, tensor_to_ndarray
from .servicer import MasterServicer
from .task_queue import _TaskQueue

class TestModel(tf.keras.Model):

    def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense_1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense_1(inputs)
        return self.dense_2(x)

    @staticmethod
    def input_shapes():
        return (10, 10)

    @staticmethod
    def optimizer(lr=0.1):
        return tf.train.GradientDescentOptimizer(lr)

class ServicerTest(unittest.TestCase):
    def testGetEmptyTask(self):
        master = MasterServicer(
            logging.getLogger(),
            2,
            3,
            None,
            _TaskQueue({}, record_per_task=3, num_epoch=2),
        )

        # No task yet, make sure the returned versions are as expected.
        task = master.GetTask(empty_pb2.Empty(), None)
        self.assertEqual("", task.shard_file_name)
        self.assertEqual(0, task.model_version)

        master._version = 1
        task = master.GetTask(empty_pb2.Empty(), None)
        self.assertEqual("", task.shard_file_name)
        self.assertEqual(1, task.model_version)

    def testGetModel(self):
        master = MasterServicer(logging.getLogger(), 2, 3, None, None)
        req = master_pb2.GetModelRequest()
        req.min_version = 0

        # Get version 0
        master.set_model_var("x", np.array([1.0, 1.0], dtype=np.float32))
        model = master.GetModel(req, None)
        self.assertEqual(0, model.version)
        self.assertEqual(["x"], list(model.param.keys()))
        np.testing.assert_array_equal(
            np.array([1.0, 1.0]), tensor_to_ndarray(model.param["x"])
        )

        # increase master's model version, now should get version 1
        master._version = 1
        master.set_model_var("x", np.array([2.0, 2.0], dtype=np.float32))
        master.set_model_var("y", np.array([12.0, 13.0], dtype=np.float32))
        model = master.GetModel(req, None)
        self.assertEqual(1, model.version)
        self.assertEqual(["x", "y"], list(model.param.keys()))
        np.testing.assert_array_equal(
            np.array([2.0, 2.0]), tensor_to_ndarray(model.param["x"])
        )
        np.testing.assert_array_equal(
            np.array([12.0, 13.0]), tensor_to_ndarray(model.param["y"])
        )

        # try to get version 2, it should raise exception.
        req.min_version = 2
        self.assertRaises(ValueError, master.GetModel, req, None)

    def testReportGradient(self):
        def makeGrad():
            """ Make a ReportTaskResultRequest compatible with model"""
            req = master_pb2.ReportGradientRequest()
            req.gradient["x"].CopyFrom(
                ndarray_to_tensor(np.array([0.1], dtype=np.float32))
            )
            req.gradient["y"].CopyFrom(
                ndarray_to_tensor(np.array([0.03, 0.06], dtype=np.float32))
            )
            req.model_version = 1
            return req

        master = MasterServicer(
            logging.getLogger(),
            3,
            3,
            tf.train.GradientDescentOptimizer(0.1),
            None,
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
            {"shard_1": 10, "shard_2": 9}, record_per_task=3, num_epoch=2
        )
        master = MasterServicer(logging.getLogger(), 3, 3, None, task_q)

        # task to number of runs.
        tasks = defaultdict(int)
        while True:
            task = master.GetTask(empty_pb2.Empty(), None)
            if not task.shard_file_name:
                break
            task_key = (task.shard_file_name, task.start, task.end)
            tasks[task_key] += 1
            report = master_pb2.ReportTaskResultRequest()
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
            }, tasks
        )

    def testUserDefinedModel(self):
        master = MasterServicer(logging.getLogger(), 2, 3, None, None)
        req = master_pb2.GetModelRequest()
        req.min_version = 0

        # Get version 0
        model_inst = TestModel()
        model_inst.build(TestModel.input_shapes())
        for variable in model_inst.trainable_variables:
            master.set_model_var(variable.name, variable.numpy())
        model = master.GetModel(req, None)
        self.assertEqual(0, model.version)
        self.assertEqual(['dense/bias:0', 'dense/kernel:0', 'dense_1/bias:0', 
            'dense_1/kernel:0'], list(sorted(model.param.keys())))
