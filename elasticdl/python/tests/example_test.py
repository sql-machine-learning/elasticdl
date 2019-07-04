import os
import tempfile
import unittest
from contextlib import closing

import numpy as np
import recordio
import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.model_helper import get_model_file
from elasticdl.python.master.checkpoint_service import CheckpointService
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.master.task_dispatcher import _TaskDispatcher
from elasticdl.python.tests.in_process_master import InProcessMaster
from elasticdl.python.worker.worker import Worker


def _get_model_info(file_name):
    module_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../examples", file_name
    )
    return module_file


def create_recordio_file(size, shape):
    image_size = 1
    for s in shape:
        image_size *= s
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with closing(recordio.Writer(temp_file.name)) as f:
        for _ in range(size):
            image = np.random.rand(image_size).astype(np.float32)
            label = np.ndarray([1], dtype=np.int64)
            label[0] = np.random.randint(0, 10)
            example_dict = {
                "image": tf.train.Feature(
                    float_list=tf.train.FloatList(value=image)
                ),
                "label": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[label])
                ),
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=example_dict)
            )
            f.write(example.SerializeToString())
    return temp_file.name


class ExampleTest(unittest.TestCase):
    def distributed_train_and_evaluate(
        self, model_def, image_shape, training=True
    ):
        """
        Run distributed training and evaluation with a local master.
        grpc calls are mocked by local master call.
        """
        module_file = _get_model_info(get_model_file(model_def))

        worker = Worker(1, module_file, None)

        shards = {create_recordio_file(128, image_shape): 128}
        if training:
            training_shards = shards
            evaluation_shards = {}
        else:
            training_shards = {}
            evaluation_shards = shards
        task_d = _TaskDispatcher(
            training_shards,
            evaluation_shards,
            {},
            records_per_task=64,
            num_epochs=1,
        )
        master = MasterServicer(
            2,
            16,
            worker._opt_fn(),
            task_d,
            init_var=[],
            checkpoint_filename_for_init="",
            checkpoint_service=CheckpointService("", 0, 0, True),
            evaluation_service=None,
        )
        worker._stub = InProcessMaster(master)

        for var in worker._model.trainable_variables:
            master.set_model_var(var.name, var.numpy())

        worker.run()

        req = elasticdl_pb2.GetTaskRequest()
        req.worker_id = 1
        task = master.GetTask(req, None)
        # No more task.
        self.assertTrue(not task.shard_file_name)

    def test_mnist_functional_train(self):
        self.distributed_train_and_evaluate(
            "mnist_functional_api", [28, 28], training=True
        )

    def test_mnist_functional_evaluate(self):
        self.distributed_train_and_evaluate(
            "mnist_functional_api", [28, 28], training=False
        )

    def test_mnist_subclass_train(self):
        self.distributed_train_and_evaluate(
            "mnist_subclass", [28, 28], training=True
        )

    def test_mnist_subclass_evaluate(self):
        self.distributed_train_and_evaluate(
            "mnist_subclass", [28, 28], training=False
        )

    def test_cifar10_functional_train(self):
        self.distributed_train_and_evaluate(
            "cifar10_functional_api", [32, 32, 3], training=True
        )

    def test_cifar10_functional_evaluate(self):
        self.distributed_train_and_evaluate(
            "cifar10_functional_api", [32, 32, 3], training=False
        )

    def test_cifar10_subclass_train(self):
        self.distributed_train_and_evaluate(
            "cifar10_subclass", [32, 32, 3], training=True
        )

    def test_cifar10_subclass_evaluate(self):
        self.distributed_train_and_evaluate(
            "cifar10_subclass", [32, 32, 3], training=False
        )

    def test_resnet50_subclass_evaluate(self):
        self.distributed_train_and_evaluate(
            "resnet50", [224, 224, 3], training=False
        )


if __name__ == "__main__":
    unittest.main()
