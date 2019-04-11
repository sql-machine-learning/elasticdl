import tensorflow as tf
tf.enable_eager_execution()

from master.task_queue import _TaskQueue
from master.servicer import MasterServicer
from proto import master_pb2_grpc
from proto import master_pb2
from .worker import Worker
import os
import logging
import tempfile
import mock
import grpc
import unittest
import numpy as np
import recordio


def input_fn(kwargs):
    def gen():
        for i in range(64):
            x = np.random.rand((1)).astype(np.float32)
            y = np.float32(2 * x + 1)
            yield {'x': x, 'y': y}

    dataset = tf.data.Dataset.from_generator(
        gen, output_types={'x': tf.float32, 'y': tf.float32},
        output_shapes={'x': tf.TensorShape([1]), 'y': tf.TensorShape([1])})

    return dataset


def batch_input_fn(records):
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


def get_optimizer(lr=0.1):
    return tf.train.GradientDescentOptimizer(lr)


class TestModel(object):
    def __init__(self):
        input1 = tf.keras.layers.Input(shape=(1,))
        x1 = tf.keras.layers.Dense(1)(input1)
        self._model = tf.keras.models.Model(input1, x1)

    def get_keras_model(self):
        return self._model

    def output(self, data):
        return self._model.call(data['x'])

    def loss(self, output, data):
        return tf.reduce_mean(tf.square(output - data['y']))


class WorkerTest(unittest.TestCase):
    def test_local_train(self):
        worker = Worker(TestModel, input_fn, get_optimizer)
        batch_size = 32
        epoch = 2
        try:
            worker.local_train(batch_size, epoch)
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

        def create_recordio_file(size):
            temp_file = tempfile.mkstemp()
            os.close(temp_file[0])
            with recordio.File(temp_file[1], 'w', max_chunk_size=size) as f:
                for _ in range(size):
                    x = np.random.rand((1)).astype(np.float32)
                    y = 2 * x + 1
                    data = np.concatenate((x, y), axis=None).tobytes()
                    f.write(data)
            return temp_file[1]

        def mock_GetTask(req):
            return master.GetTask(req, None)

        def mock_GetModel(req):
            return master.GetModel(req, None)

        def mock_ReportGradient(req):
            return master.ReportGradient(req, None)

        def mock_ReportTaskResult(req):
            return master.ReportTaskResult(req, None)

        channel = grpc.insecure_channel('localhost:9999')
        worker = Worker(TestModel, batch_input_fn, get_optimizer, channel)

        filename = create_recordio_file(128)
        task_q = _TaskQueue(
            {filename: 128}, record_per_task=64, num_epoch=1
        )
        master = MasterServicer(logging.getLogger(),
                                2,
                                16,
                                get_optimizer(),
                                task_q)
        for var in worker._keras_model.variables:
            master._set_model_var(Worker.replaced_name(var.name), var.numpy())

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
