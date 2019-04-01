import logging
import unittest

from google.protobuf import empty_pb2

from proto import master_pb2
from .servicer import MasterServicer


class ServicerTest(unittest.TestCase):
    def testGetTask(self):
        master = MasterServicer(logging.getLogger())

        # No task yet, make sure the returned versions are as expected.
        task = master.GetTask(empty_pb2.Empty(), None)
        self.assertEqual("", task.shard_file_name)
        self.assertEqual(0, task.model_version)

        master._version = 1
        task = master.GetTask(empty_pb2.Empty(), None)
        self.assertEqual("", task.shard_file_name)
        self.assertEqual(1, task.model_version)

    def testGetModel(self):
        master = MasterServicer(logging.getLogger())
        req = master_pb2.GetModelRequest()
        req.min_version = 0

        # Get version 0
        model = master.GetModel(req, None)
        self.assertEqual(0, model.version)

        # increase master's model version, now should get version 1
        master._version = 1
        model = master.GetModel(req, None)
        self.assertEqual(1, model.version)

        # try to get version 2, it should raise exception.
        req.min_version = 2
        self.assertRaises(ValueError, master.GetModel, req, None)


    def testReportTaskResult(self):
        master = MasterServicer(logging.getLogger())
        master._version = 1
        req = master_pb2.ReportTaskResultRequest()

        # Report a future version, should raise exception
        req.model_version = 2
        self.assertRaises(ValueError, master.ReportTaskResult, req, None)

        # Report an old version, should not be accepted
        req.model_version = 0
        res = master.ReportTaskResult(req, None)
        self.assertFalse(res.accepted)
        self.assertEqual(1, res.model_version)

        # Report a current version, but with error, should not be accepted.
        req.model_version = 1
        req.err_message = "worker error"
        res = master.ReportTaskResult(req, None)
        self.assertFalse(res.accepted)
        self.assertEqual(1, res.model_version)

        # Report a current version without error, should be accepted.
        req.model_version = 1
        req.err_message = ""
        res = master.ReportTaskResult(req, None)
        self.assertTrue(res.accepted)
        self.assertEqual(1, res.model_version)
