import logging
import unittest
import numpy as np

import tensorflow as tf

tf.enable_eager_execution()

from google.protobuf import empty_pb2

from proto import master_pb2
from util.ndarray import ndarray_to_tensor, tensor_to_ndarray
from .servicer import MasterServicer


class ServicerTest(unittest.TestCase):
    def testGetTask(self):
        master = MasterServicer(logging.getLogger(), 2, None)

        # No task yet, make sure the returned versions are as expected.
        task = master.GetTask(empty_pb2.Empty(), None)
        self.assertEqual("", task.shard_file_name)
        self.assertEqual(0, task.model_version)

        master._version = 1
        task = master.GetTask(empty_pb2.Empty(), None)
        self.assertEqual("", task.shard_file_name)
        self.assertEqual(1, task.model_version)

    def testGetModel(self):
        master = MasterServicer(logging.getLogger(), 2, None)
        req = master_pb2.GetModelRequest()
        req.min_version = 0

        # Get version 0
        master._set_model_var("x", np.array([1.0, 1.0], dtype=np.float32))
        model = master.GetModel(req, None)
        self.assertEqual(0, model.version)
        self.assertEqual(["x"], list(model.param.keys()))
        np.testing.assert_array_equal(
            np.array([1.0, 1.0]), tensor_to_ndarray(model.param["x"])
        )

        # increase master's model version, now should get version 1
        master._version = 1
        master._set_model_var("x", np.array([2.0, 2.0], dtype=np.float32))
        master._set_model_var("y", np.array([12.0, 13.0], dtype=np.float32))
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
            logging.getLogger(), 3, tf.train.GradientDescentOptimizer(0.1)
        )
        master._version = 1
        master._set_model_var("x", np.array([2.0], dtype=np.float32))
        master._set_model_var("y", np.array([12.0, 13.0], dtype=np.float32))

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

        # Report a current version without error, should be accepted.
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

        # Report a current version without error, should be accepted, and a new
        # version created
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
        # TODO: add tests and verify task queue
        pass
