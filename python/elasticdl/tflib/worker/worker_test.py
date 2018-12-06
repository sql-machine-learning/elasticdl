import unittest
import threading
import queue
from unittest.mock import patch
from recordio.file_index import _ChunkData as C
from recordio.file import File
from elasticdl.tflib import ParameterServerClient
from elasticdl.tflib import ParameterServer
from elasticdl.system.master import Master
from elasticdl.tflib import Worker
import tensorflow as tf
import numpy as np


class MockRecordIoFile(object):
    def __init__(self, index):
        self._index = index

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def get_index(self):
        return self._index


class MockMasterThread(threading.Thread):
    def __init__(self):
        self.recordio_data = {'f0': [C(0, 100, 2), C(200, 100, 3)], 'f1': [
            C(10, 200, 4), C(210, 200, 4)]}
        self.master = Master(
            self.recordio_data.keys(),
            num_epoch=3,
            max_trial=2)
        threading.Thread.__init__(self)

    def run(self):
        # patch Master's recordio calls to inject mock data
        print("start running patched master ")
        with patch('elasticdl.system.master.File', autospec=True) as mock:
            mock.side_effect = [MockRecordIoFile(
                index) for index in self.recordio_data.values()]
            self.master.run()

    def register_worker(self):
        return self.master.register_worker()


class Dummy(object):
    # This should be called first such that optimizer() and vars() would return valid values.
    def __init__(self):
        self._opt = tf.train.GradientDescentOptimizer(0.1)
        self._vars = None

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._W = tf.get_variable("x", [1])
            self._b = tf.get_variable("y", [1])

    def optimizer(self):
        return self._opt

    @staticmethod
    def _create_model():
        input_shape = [1]

        l = tf.keras.layers
        return tf.keras.Sequential(
            [
                l.Reshape(target_shape=input_shape,
                          input_shape=(1,)),
                l.Dense(1, input_shape=input_shape)
            ])

    def _init_vars(self):
        graph = tf.Graph()
        with graph.as_default():
            self._create_model()
            trainable_vars = tf.trainable_variables()
            init_op = tf.initializers.global_variables()
        # strip the variable name  part of ':0'
        var_names = [v.name.split(":", 1)[0] for v in trainable_vars]
        with tf.Session(graph=graph) as sess:
            sess.run(init_op)
            var_values = sess.run(trainable_vars)
        self._vars = dict(zip(var_names, var_values))

    def vars(self):
        if self._vars is None:
            self._init_vars()
        return self._vars

    @staticmethod
    def forward(x):
        model = Dummy._create_model()
        return model(x)

    @staticmethod
    def loss(y_predict, y_true):
        return tf.reduce_mean(tf.square(y_true - y_predict))


class WorkerTestCase(unittest.TestCase):
    def test(self):
        prog = Dummy()

        ps_num = 1
        worker_num = 2

        ps = [ParameterServer(prog.optimizer(), prog.vars())
              for _ in range(ps_num)]
        for p in ps:
            p.start()

        m = MockMasterThread()
        m.start()

        worker = [Worker(ps_configs=ps, master=m,
                         forward_func=prog.forward,
                         loss_func=prog.loss,
                         optimizer=prog.optimizer())
                  for _ in range(worker_num)]
        for w in worker:
            w.start()

        m.join()
        for w in worker:
            w.join()
        for p in ps:
            p.join()


if __name__ == '__main__':
    unittest.main()
