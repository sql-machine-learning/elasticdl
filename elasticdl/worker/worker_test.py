import recordio
import os
import tempfile
import mock
import unittest
import numpy as np

import tensorflow as tf
tf.enable_eager_execution()

from .worker import Worker
from proto import master_pb2


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


"""
Mock the methods in Worker class, so that we can test Worker.distributed_train locally.
get_task: mock Worker.get_task.
          Return a task for a generated recordio file in the first call, empty task in the second call.
get_model: mock Worker.get_model.
           Do nothing and keep the local model.
report_gradient: mock Worker.report_gradient.
                 Update local model with the gradients.
report_task_result: mock Worker.report_task_result.
                    Print the task result.
"""


def create_task(size, batch_size):
    temp_file = tempfile.mkstemp()
    os.close(temp_file[0])
    with recordio.File(temp_file[1], 'w', max_chunk_size=size) as f:
        for _ in range(size):
            x = np.random.rand((1)).astype(np.float32)
            y = 2 * x + 1
            data = np.concatenate((x, y), axis=None).tobytes()
            f.write(data)

    task = master_pb2.Task()
    task.task_id = 0
    task.minibatch_size = batch_size
    task.shard_file_name = temp_file[1]
    task.start = 0
    task.end = size - 1
    task.model_version = 0

    empty_task = master_pb2.Task()
    return [task, empty_task]


tasks = create_task(128, 16)
opt = get_optimizer()


def get_task():
    """
    mock Worker.get_task
    """
    return tasks.pop(0)


def get_model(min_version):
    """
    mock Worker.get_model
    """
    pass


def report_gradient(grads, variables):
    """
    mock Worker.report_gradient
    """
    # Update local model with the gradients.
    opt.apply_gradients(zip(grads, variables))


def report_task_result(task_id, err_msg):
    """
    mock Worker.report_task_result
    """
    if not err_msg:
        print('Task %d finished successfully.' % task_id)
    else:
        print('Task %d failed: %s' % (task_id, err_msg))
    pass


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

    @mock.patch(Worker.__module__ + '.Worker.get_task', side_effect=get_task)
    @mock.patch(Worker.__module__ + '.Worker.get_model', side_effect=get_model)
    @mock.patch(Worker.__module__ + '.Worker.report_gradient', side_effect=report_gradient)
    @mock.patch(Worker.__module__ + '.Worker.report_task_result', side_effect=report_task_result)
    def test_distributed_train(self, mock_report_task_result, mock_report_gradient, mock_get_model, mock_get_task):
        worker = Worker(TestModel, batch_input_fn, get_optimizer)
        try:
            worker.distributed_train()
            res = True
        except Exception as ex:
            print(ex)
            res = False
        self.assertTrue(res)
