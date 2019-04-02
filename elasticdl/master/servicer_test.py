import logging
import unittest
import numpy as np

from google.protobuf import empty_pb2

from proto import master_pb2
from util.converter import TensorToNdarray, NdarrayToTensor
from .servicer import MasterServicer


class ServicerTest(unittest.TestCase):
    def testGetTask(self):
        master = MasterServicer(logging.getLogger(), 2)

        # No task yet, make sure the returned versions are as expected.
        task = master.GetTask(empty_pb2.Empty(), None)
        self.assertEqual("", task.shard_file_name)
        self.assertEqual(0, task.model_version)

        master._version = 1
        task = master.GetTask(empty_pb2.Empty(), None)
        self.assertEqual("", task.shard_file_name)
        self.assertEqual(1, task.model_version)

    def testGetModel(self):
        master = MasterServicer(logging.getLogger(), 2)
        req = master_pb2.GetModelRequest()
        req.min_version = 0

        # Get version 0
        master._model["x"] = np.array([1.0, 1.0], dtype=np.float32)
        model = master.GetModel(req, None)
        self.assertEqual(0, model.version)
        self.assertEqual(["x"], list(model.param.keys()))
        np.testing.assert_array_equal(
            np.array([1.0, 1.0]), TensorToNdarray(model.param["x"])
        )

        # increase master's model version, now should get version 1
        master._version = 1
        master._model["x"] = np.array([2.0, 2.0], dtype=np.float32)
        master._model["y"] = np.array([12.0, 13.0], dtype=np.float32)
        model = master.GetModel(req, None)
        self.assertEqual(1, model.version)
        self.assertEqual(["x", "y"], list(model.param.keys()))
        np.testing.assert_array_equal(
            np.array([2.0, 2.0]), TensorToNdarray(model.param["x"])
        )
        np.testing.assert_array_equal(
            np.array([12.0, 13.0]), TensorToNdarray(model.param["y"])
        )

        # try to get version 2, it should raise exception.
        req.min_version = 2
        self.assertRaises(ValueError, master.GetModel, req, None)

    def testReportTaskResult(self):
        def makeGrad():
            """ Make a ReportTaskResultRequest compatible with model"""
            req = master_pb2.ReportTaskResultRequest()
            req.gradient["x"].CopyFrom(
                NdarrayToTensor(np.array([0.1], dtype=np.float32))
            )
            req.gradient["y"].CopyFrom(
                NdarrayToTensor(np.array([0.1, 0.2], dtype=np.float32))
            )
            req.model_version = 1
            return req

        master = MasterServicer(logging.getLogger(), 3)
        master._version = 1
        master._model["x"] = np.array([2.0], dtype=np.float32)
        master._model["y"] = np.array([12.0, 13.0], dtype=np.float32)

        # Report a future version, should raise exception
        req = makeGrad()
        req.model_version = 2
        self.assertRaises(ValueError, master.ReportTaskResult, req, None)

        # Report an old version, should not be accepted
        req = makeGrad()
        req.model_version = 0
        res = master.ReportTaskResult(req, None)
        self.assertFalse(res.accepted)
        self.assertEqual(1, res.model_version)

        # Report a current version, but with error, should not be accepted.
        req = makeGrad()
        req.err_message = "worker error"
        res = master.ReportTaskResult(req, None)
        self.assertFalse(res.accepted)
        self.assertEqual(1, res.model_version)

        # Report a unknown gradient, should raise.
        req = makeGrad()
        req.gradient["z"].CopyFrom(
            NdarrayToTensor(np.array([0.1], dtype=np.float32))
        )
        self.assertRaises(ValueError, master.ReportTaskResult, req, None)

        # Report an incompatible gradient, should raise.
        req = makeGrad()
        req.gradient["y"].CopyFrom(
            NdarrayToTensor(np.array([0.1], dtype=np.float32))
        )
        self.assertRaises(ValueError, master.ReportTaskResult, req, None)

        # Report a current version without error, should be accepted.
        req = makeGrad()
        res = master.ReportTaskResult(req, None)
        self.assertTrue(res.accepted)
        self.assertEqual(1, res.model_version)

        # Report a current version with part of gradients, should be accepted.
        req = makeGrad()
        del req.gradient["y"]
        res = master.ReportTaskResult(req, None)
        self.assertTrue(res.accepted)
        self.assertEqual(1, res.model_version)
        # Gradient should be accumulated.
        np.testing.assert_array_equal(
            np.array([0.2], dtype=np.float32), master._gradient_sum["x"]
        )
        np.testing.assert_array_equal(
            np.array([0.1, 0.2], dtype=np.float32), master._gradient_sum["y"]
        )
        self.assertEqual(2, master._grad_n)

        # Report a current version without error, should be accepted, and a new
        # version created
        req = makeGrad()
        res = master.ReportTaskResult(req, None)
        self.assertTrue(res.accepted)
        self.assertEqual(2, res.model_version)
        # TODO: verify model when model updating is in place.
        self.assertFalse(master._gradient_sum)
        self.assertEqual(0, master._grad_n)
