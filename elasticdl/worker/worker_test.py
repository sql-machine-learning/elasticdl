import tensorflow as tf
tf.enable_eager_execution()

from master.task_queue import _TaskQueue
from master.servicer import MasterServicer
from google.protobuf import empty_pb2
from proto import master_pb2_grpc
from proto import master_pb2
from .worker import Worker
import logging
import tempfile
import mock
import grpc
import unittest
import numpy as np
import recordio


class TestModel(tf.keras.Model):
    def __init__(self):
        super(TestModel, self).__init__(name='test_model')
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        return self.dense(inputs)

    @staticmethod
    def input_shapes():
        return (1, 1)

    @staticmethod
    def input_names():
        return ['x']

    @staticmethod
    def loss(outputs, labels):
        return tf.reduce_mean(tf.square(outputs - labels['y'])) 

    @staticmethod
    def input_fn(records):
        x_list = []
        y_list = []
        # deserialize
        for r in records:
            parsed = np.frombuffer(r, dtype='float32')
            x_list.append([parsed[0]])
            y_list.append([parsed[1]])
        # batching
        batch_size = len(x_list)
        xs = np.concatenate(x_list, axis=0)
        xs = np.reshape(xs, (batch_size, 1))
        ys = np.concatenate(y_list, axis=0)
        ys = np.reshape(xs, (batch_size, 1))
        return {'x': xs, 'y': ys}

    @staticmethod
    def optimizer(lr=0.1):
        return tf.train.GradientDescentOptimizer(lr)


def create_recordio_file(size):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with recordio.File(temp_file.name, 'w', max_chunk_size=size) as f:
        for _ in range(size):
            x = np.random.rand((1)).astype(np.float32)
            y = 2 * x + 1
            data = np.concatenate((x, y), axis=None).tobytes()
            f.write(data)
    return temp_file.name

class WorkerTest(unittest.TestCase):
    def test_local_train(self):
        worker = Worker(TestModel)
        filename = create_recordio_file(128)
        batch_size = 32
        epoch = 2
        try:
            worker.local_train([filename], batch_size, epoch)
            res = True
        except Exception as ex:
            print(ex)
            res = False
        self.assertTrue(res)

    def test_distributed_train(self):
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
        worker = Worker(TestModel, channel)

        filename = create_recordio_file(128)
        task_q = _TaskQueue(
            {filename: 128}, record_per_task=64, num_epoch=1
        )
        master = MasterServicer(logging.getLogger(),
                                2,
                                16,
                                TestModel.optimizer(),
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
