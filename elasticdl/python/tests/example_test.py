import tensorflow as tf

tf.enable_eager_execution()

import tempfile
import mock
import grpc
import os
import unittest
import numpy as np
import recordio

from contextlib import closing
from elasticdl.proto import elasticdl_pb2
from elasticdl.python.elasticdl.common.model_helper import load_user_model
from elasticdl.python.elasticdl.master.task_queue import _TaskQueue
from elasticdl.python.elasticdl.master.servicer import MasterServicer
from elasticdl.python.elasticdl.worker.worker import Worker
from elasticdl.python.data.codec import BytesCodec
from elasticdl.python.data.codec import TFExampleCodec


def _get_model_info(file_name):
    module_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../examples", file_name
    )
    m = load_user_model(module_file)
    columns = m.feature_columns() + m.label_columns()

    return module_file, columns


def create_recordio_file(size, shape, codec_type, columns):
    codec = None
    if codec_type == "bytes":
        codec = BytesCodec(columns)
    elif codec_type == "tf_example":
        codec = TFExampleCodec(columns)

    image_size = 1
    for s in shape:
        image_size *= s
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with closing(recordio.Writer(temp_file.name)) as f:
        for _ in range(size):
            image = np.random.rand(image_size).astype(np.float32).reshape(shape)
            label = np.ndarray([1], dtype=np.int64)
            label[0] = np.random.randint(0, 10)
            f.write(codec.encode({"image": image, "label": label}))
    return temp_file.name


class ExampleTest(unittest.TestCase):
    def distributed_train_and_evaluate(
        self, file_name, codec_type, image_shape, training=True
    ):
        """
        Run distributed training and evaluation with a local master.
        grpc calls are mocked by local master call.
        """

        def mock_GetTask(req):
            return master.GetTask(req, None)

        def mock_GetModel(req):
            return master.GetModel(req, None)

        def mock_ReportGradient(req):
            return master.ReportGradient(req, None)

        def mock_ReportEvaluationMetrics(req):
            return master.ReportEvaluationMetrics(req, None)

        def mock_ReportTaskResult(req):
            return master.ReportTaskResult(req, None)

        module_file, columns = _get_model_info(file_name)

        channel = grpc.insecure_channel("localhost:9999")
        worker = Worker(1, module_file, channel, codec_type=codec_type)

        shards = {create_recordio_file(128, image_shape, codec_type, columns): 128}
        if training:
            training_shards = shards
            evaluation_shards = {}
        else:
            training_shards = {}
            evaluation_shards = shards
        task_q = _TaskQueue(
            training_shards, evaluation_shards, records_per_task=64, num_epochs=1
        )
        master = MasterServicer(
            2,
            16,
            worker._opt_fn(),
            task_q,
            init_var=[],
            init_from_checkpoint="",
            checkpoint_dir="",
            checkpoint_steps=0,
            keep_checkpoint_max=0,
        )

        for var in worker._model.trainable_variables:
            master.set_model_var(var.name, var.numpy())

        with mock.patch.object(
            worker._stub, "GetTask", mock_GetTask
        ), mock.patch.object(
            worker._stub, "GetModel", mock_GetModel
        ), mock.patch.object(
            worker._stub, "ReportGradient", mock_ReportGradient
        ), mock.patch.object(
            worker._stub, "ReportEvaluationMetrics", mock_ReportEvaluationMetrics
        ), mock.patch.object(
            worker._stub, "ReportTaskResult", mock_ReportTaskResult
        ):
            try:
                worker.run()
                res = True
            except Exception as ex:
                print(ex)
                res = False

        self.assertTrue(res)
        req = elasticdl_pb2.GetTaskRequest()
        req.worker_id = 1
        task = mock_GetTask(req)
        # No more task.
        self.assertTrue(not task.shard_file_name)

    def test_mnist_functional_bytes_train(self):
        self.distributed_train_and_evaluate(
            "mnist_functional_api.py", "bytes", [28, 28], training=True
        )

    def test_mnist_functional_bytes_evaluate(self):
        self.distributed_train_and_evaluate(
            "mnist_functional_api.py", "bytes", [28, 28], training=False
        )

    def test_mnist_functional_tfexample_train(self):
        self.distributed_train_and_evaluate(
            "mnist_functional_api.py", "tf_example", [28, 28], training=True
        )

    def test_mnist_functional_tfexample_evaluate(self):
        self.distributed_train_and_evaluate(
            "mnist_functional_api.py", "tf_example", [28, 28], training=False
        )

    def test_mnist_subclass_bytes_train(self):
        self.distributed_train_and_evaluate(
            "mnist_subclass.py", "bytes", [28, 28], training=True
        )

    def test_mnist_subclass_bytes_evaluate(self):
        self.distributed_train_and_evaluate(
            "mnist_subclass.py", "bytes", [28, 28], training=False
        )

    def test_mnist_subclass_tfexample_train(self):
        self.distributed_train_and_evaluate(
            "mnist_subclass.py", "tf_example", [28, 28], training=True
        )

    def test_mnist_subclass_tfexample_evaluate(self):
        self.distributed_train_and_evaluate(
            "mnist_subclass.py", "tf_example", [28, 28], training=False
        )

    def test_cifar10_functional_bytes_train(self):
        self.distributed_train_and_evaluate(
            "cifar10_functional_api.py", "bytes", [32, 32, 3], training=True
        )

    def test_cifar10_functional_bytes_evaluate(self):
        self.distributed_train_and_evaluate(
            "cifar10_functional_api.py", "bytes", [32, 32, 3], training=False
        )

    def test_cifar10_functional_tfexample_train(self):
        self.distributed_train_and_evaluate(
            "cifar10_functional_api.py", "tf_example", [32, 32, 3], training=True
        )

    def test_cifar10_functional_tfexample_evaluate(self):
        self.distributed_train_and_evaluate(
            "cifar10_functional_api.py", "tf_example", [32, 32, 3], training=False
        )

    def test_cifar10_subclass_bytes_train(self):
        self.distributed_train_and_evaluate(
            "cifar10_subclass.py", "bytes", [32, 32, 3], training=True
        )

    def test_cifar10_subclass_bytes_evaluate(self):
        self.distributed_train_and_evaluate(
            "cifar10_subclass.py", "bytes", [32, 32, 3], training=False
        )

    def test_cifar10_subclass_tfexample_train(self):
        self.distributed_train_and_evaluate(
            "cifar10_subclass.py", "tf_example", [32, 32, 3], training=True
        )

    def test_cifar10_subclass_tfexample_evaluate(self):
        self.distributed_train_and_evaluate(
            "cifar10_subclass.py", "tf_example", [32, 32, 3], training=False
        )


if __name__ == "__main__":
    unittest.main()
