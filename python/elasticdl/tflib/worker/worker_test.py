import unittest
import threading
import queue
from unittest.mock import patch
from recordio.file_index import _ChunkData as C
from recordio.file import File
from elasticdl.tflib import ParameterServerClient, no_partition, ParameterServer, Worker
from elasticdl.system.master import Master
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
        self.recordio_data = {
            "f0": [C(0, 100, 2), C(200, 100, 3)],
            "f1": [C(10, 200, 4), C(210, 200, 4)],
        }
        self.master = Master(self.recordio_data.keys(), num_epoch=3, max_trial=2)
        threading.Thread.__init__(self)

    def run(self):
        # patch Master's recordio calls to inject mock data
        print("start running patched master ")
        with patch("elasticdl.system.master.File", autospec=True) as mock:
            mock.side_effect = [
                MockRecordIoFile(index) for index in self.recordio_data.values()
            ]
            self.master.run()

    def register_worker(self):
        return self.master.register_worker()


class Dummy(object):
    # This should be called first such that optimizer() and vars() would return valid values.
    def __init__(self):
        self._vars = None

    @staticmethod
    def optimizer():
        return tf.train.GradientDescentOptimizer(0.1)

    @staticmethod
    def _create_model_var():
        var_dict = {}
        var_dict["x"] = tf.get_variable("x", [1])
        var_dict["y"] = tf.get_variable("y", [1])
        return var_dict

    def _init_vars(self):
        graph = tf.Graph()
        with graph.as_default():
            self._create_model_var()
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
        model_var = Dummy._create_model_var()
        return model_var["x"] * x + model_var["y"]

    @staticmethod
    def loss(y_predict, y_true):
        return tf.reduce_mean(tf.square(y_true - y_predict))

    @staticmethod
    def data_process_py_func(data):
        parsed = np.frombuffer(data, dtype="float32")
        x = parsed[0]
        y = parsed[1]
        return x, y

    # data_process_py_func output data type
    data_py_output_type = [tf.float32, tf.float32]

    # data_process_tf_func is optional. Add it to verify the functionality.
    @staticmethod
    def data_process_tf_func(x, y):
        _x = tf.identity(x)
        _y = tf.identity(y)
        return x, y


def dummy_create_recordio_dataset(data_file, file_offset):
    def gen():
        for i in range(200):
            x = np.float32(np.random.rand())
            y = np.float32(2 * x + 1)
            yield np.concatenate((x, y), axis=None).tobytes()

    dataset = tf.data.Dataset.from_generator(gen, (tf.string), (tf.TensorShape([])))

    return dataset


@patch.object(Worker, "_create_recordio_dataset", dummy_create_recordio_dataset)
class WorkerTestCase(unittest.TestCase):
    def test(self):
        prog = Dummy()

        ps_num = 1
        worker_num = 2

        ps = [ParameterServer(prog.optimizer(), prog.vars()) for _ in range(ps_num)]
        for p in ps:
            p.start()

        ps_client = ParameterServerClient(ps_configs=ps, partition_func=no_partition)

        m = MockMasterThread()
        m.start()

        worker = [
            Worker(ps_client=ps_client, work_queue=m.register_worker(), umd=Dummy)
            for _ in range(worker_num)
        ]
        for w in worker:
            w.start()

        m.join()
        for w in worker:
            w.join()

        base_step, var_values = ps_client.pull()
        print("Weights after training step %d : " % base_step, var_values)

        for p in ps:
            p.join()


if __name__ == "__main__":
    unittest.main()
