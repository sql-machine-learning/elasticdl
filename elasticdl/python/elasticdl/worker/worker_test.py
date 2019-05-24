import tensorflow as tf
tf.enable_eager_execution()

import logging
import tempfile
import mock
import grpc
import os
import unittest
import numpy as np
import recordio

from elasticdl.common.model_helper import load_user_model
from elasticdl.master.task_queue import _TaskQueue
from elasticdl.master.servicer import MasterServicer
from google.protobuf import empty_pb2
from elasticdl.proto import master_pb2_grpc
from elasticdl.proto import master_pb2
from elasticdl.worker.worker import Worker
from data.codec import BytesCodec
from data.codec import TFExampleCodec

_module_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "test_module.py")

m = load_user_model(_module_file)
columns = m.feature_columns() + m.label_columns()

def create_recordio_file(size, codec_type):
    codec = None
    if codec_type == "bytes":
        codec = BytesCodec(columns)
    elif codec_type == "tf_example":
        codec = TFExampleCodec(columns)

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with recordio.File(temp_file.name, 'w', max_chunk_size=size, encoder=codec.encode) as f:
        for _ in range(size):
            x = np.random.rand((1)).astype(np.float32)
            y = 2 * x + 1
            f.write({"x": x, "y": y})
    return temp_file.name

class WorkerTest(unittest.TestCase):
    def local_train(self, codec_type):
        worker = Worker(0, _module_file, codec_type=codec_type)
        filename = create_recordio_file(128, codec_type)
        batch_size = 32
        epoch = 2
        try:
            worker.local_train([filename], batch_size, epoch)
            res = True
        except Exception as ex:
            print(ex)
            res = False
        self.assertTrue(res)
    
    def test_local_train_bytes(self):
        self.local_train("bytes")

    def test_local_train_tf_example(self):
        self.local_train("tf_example")

    def distributed_train(self, codec_type):
        """
        Run Worker.distributed_train with a local master.
        grpc calls are mocked by local master call.
        """
        def mock_GetTask(req):
            return master.GetTask(req, None)

        def mock_GetModel(req):
            return master.GetModel(req, None)

        def mock_ReportGradient(req):
            if master._version > 2 and master._version < 80:
                # For testing of retrain when gradient not accepted.
                # Increase master version so the gradient will not be accepted.
                master._version += 1
            return master.ReportGradient(req, None)

        def mock_ReportTaskResult(req):
            return master.ReportTaskResult(req, None)

        channel = grpc.insecure_channel('localhost:9999')
        worker = Worker(0, _module_file, channel, codec_type=codec_type)

        filename = create_recordio_file(128, codec_type)
        task_q = _TaskQueue(
            {filename: 128}, record_per_task=64, num_epoch=1
        )
        master = MasterServicer(logging.getLogger(),
                                2,
                                16,
                                worker._opt_fn(),
                                task_q)

        for var in worker._model.trainable_variables:
            master.set_model_var(var.name, var.numpy())

        with mock.patch.object(worker._stub, 'GetTask', mock_GetTask),                   \
                mock.patch.object(worker._stub, 'GetModel', mock_GetModel),              \
                mock.patch.object(worker._stub, 'ReportGradient', mock_ReportGradient),  \
                mock.patch.object(worker._stub, 'ReportTaskResult', mock_ReportTaskResult):
            try:
                worker.distributed_train()
                res = True
            except Exception as ex:
                print(ex)
                res = False

        self.assertTrue(res)
        task = mock_GetTask(empty_pb2.Empty())
        # No more task.
        self.assertTrue(not task.shard_file_name)

    def test_distributed_train_bytes(self):
        self.distributed_train("bytes")

    def test_distributed_train_tf_example(self):
        self.distributed_train("tf_example")
