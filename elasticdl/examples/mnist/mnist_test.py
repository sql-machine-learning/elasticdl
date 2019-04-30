import tensorflow as tf
tf.enable_eager_execution()

import os
import unittest
from elasticdl.worker.worker import Worker
from elasticdl.common.model_helper import load_user_model


class WorkerTest(unittest.TestCase):
    def test_mnist_local_train(self):
        data_dir = "/data/mnist/train"
        model_file = os.path.dirname(__file__) + "/mnist.py"
        model_class = "MnistModel"

        model_cls = load_user_model(model_file, model_class)

        filename = []
        for f in os.listdir(data_dir):
            p = os.path.join(data_dir, f)
            filename.append(p)
        batch_size = 64
        epoch = 1

        worker = Worker(model_cls)
        try:
            worker.local_train(filename, batch_size, epoch)
            res = True
        except Exception as ex:
            print(ex)
            res = False
        self.assertTrue(res)


if __name__ == "__main__":
    unittest.main()
