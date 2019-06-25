import tempfile
import os
import unittest
import numpy as np
import recordio

from contextlib import closing
from elasticdl.proto import elasticdl_pb2
from elasticdl.python.elasticdl.common.model_helper import load_module
from elasticdl.python.elasticdl.master.task_queue import _TaskQueue
from elasticdl.python.elasticdl.master.servicer import MasterServicer
from elasticdl.python.elasticdl.worker.worker import Worker
from elasticdl.python.data.codec import TFExampleCodec
from elasticdl.python.tests.in_process_master import InProcessMaster
from elasticdl.python.elasticdl.master.checkpoint_service import (
    CheckpointService,
)


def _get_model_info(file_name):
    module_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../examples", file_name
    )
    m = load_module(module_file)
    data_schema = m.data_schema()

    return module_file, data_schema


def create_recordio_file(size, shape, data_schema):
    codec = TFExampleCodec()
    feature_name_to_type = {
        d["name"]: d["dtype"] for d in data_schema
    }

    image_size = 1
    for s in shape:
        image_size *= s
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with closing(recordio.Writer(temp_file.name)) as f:
        for _ in range(size):
            image = (
                np.random.rand(image_size).astype(np.float32).reshape(shape)
            )
            label = np.ndarray([1], dtype=np.int64)
            label[0] = np.random.randint(0, 10)
            f.write(
                codec.encode(
                    {"image": image, "label": label}, feature_name_to_type
                )
            )
    return temp_file.name


class ExampleTest(unittest.TestCase):
    def distributed_train_and_evaluate(
        self, file_name, image_shape, training=True
    ):
        """
        Run distributed training and evaluation with a local master.
        grpc calls are mocked by local master call.
        """
        codec_file = 'elasticdl/python/data/codec/tf_example_codec.py'
        module_file, data_schema = _get_model_info(file_name)

        worker = Worker(1, module_file, None, codec_file=codec_file)

        shards = {
            create_recordio_file(128, image_shape, data_schema): 128
        }
        if training:
            training_shards = shards
            evaluation_shards = {}
        else:
            training_shards = {}
            evaluation_shards = shards
        task_q = _TaskQueue(
            training_shards,
            evaluation_shards,
            records_per_task=64,
            num_epochs=1,
        )
        master = MasterServicer(
            2,
            16,
            worker._opt_fn(),
            task_q,
            init_var=[],
            checkpoint_filename_for_init="",
            checkpoint_service=CheckpointService("", 0, 0),
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

    def test_mnist_functional_bytes_train(self):
        self.distributed_train_and_evaluate(
            "mnist_functional_api.py", [28, 28], training=True
        )

    def test_mnist_functional_bytes_evaluate(self):
        self.distributed_train_and_evaluate(
            "mnist_functional_api.py", [28, 28], training=False
        )

    def test_mnist_subclass_bytes_train(self):
        self.distributed_train_and_evaluate(
            "mnist_subclass.py", [28, 28], training=True
        )

    def test_mnist_subclass_bytes_evaluate(self):
        self.distributed_train_and_evaluate(
            "mnist_subclass.py", [28, 28], training=False
        )

    def test_cifar10_functional_bytes_train(self):
        self.distributed_train_and_evaluate(
            "cifar10_functional_api.py", [32, 32, 3], training=True
        )

    def test_cifar10_functional_bytes_evaluate(self):
        self.distributed_train_and_evaluate(
            "cifar10_functional_api.py", [32, 32, 3], training=False
        )

    def test_cifar10_subclass_bytes_train(self):
        self.distributed_train_and_evaluate(
            "cifar10_subclass.py", [32, 32, 3], training=True
        )

    def test_cifar10_subclass_bytes_evaluate(self):
        self.distributed_train_and_evaluate(
            "cifar10_subclass.py", [32, 32, 3], training=False
        )


if __name__ == "__main__":
    unittest.main()
