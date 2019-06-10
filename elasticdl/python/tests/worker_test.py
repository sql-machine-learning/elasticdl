import tensorflow as tf

tf.enable_eager_execution()

import tempfile
import os
import unittest
import numpy as np
import recordio

from .in_process_master import InProcessMaster
from contextlib import closing
from elasticdl.proto import elasticdl_pb2
from elasticdl.python.elasticdl.common.model_helper import load_user_model
from elasticdl.python.elasticdl.master.task_queue import _TaskQueue
from elasticdl.python.elasticdl.master.servicer import MasterServicer
from elasticdl.python.elasticdl.worker.worker import Worker
from elasticdl.python.data.codec import BytesCodec
from elasticdl.python.data.codec import TFExampleCodec

_module_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_module.py"
)

m = load_user_model(_module_file)
columns = m.feature_columns() + m.label_columns()


def create_recordio_file(size, codec_type):
    codec = None
    if codec_type == "bytes":
        codec = BytesCodec(columns)
    elif codec_type == "tf_example":
        codec = TFExampleCodec(columns)

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with closing(recordio.Writer(temp_file.name)) as f:
        for _ in range(size):
            x = np.random.rand((1)).astype(np.float32)
            y = 2 * x + 1
            f.write(codec.encode({"x": x, "y": y}))
    return temp_file.name


class WorkerTest(unittest.TestCase):
    def distributed_train_and_evaluate(self, codec_type, training=True):
        """
        Run distributed training and evaluation with a local master.
        grpc calls are mocked by local master call.
        """

        class _Master(InProcessMaster):
            def ReportGradient(self, req):
                if 2 < self._m._version < 80:
                    # For testing of retrain when gradient not accepted.
                    # Increase master version so the gradient will not be accepted.
                    self._m._version += 1
                return self._m.ReportGradient(req, None)

            def ReportEvaluationMetrics(self, req):
                if 2 < self._m._version < 80:
                    # For testing of evaluation retries when evaluation metrics are not accepted.
                    # Increase master version so the evaluation metrics will not be accepted.
                    self._m._version += 1
                return self._m.ReportEvaluationMetrics(req, None)

        worker = Worker(1, _module_file, None, codec_type=codec_type)

        shards = {create_recordio_file(128, codec_type): 128}
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
            init_from_checkpoint="",
            checkpoint_dir="",
            checkpoint_steps=0,
            keep_checkpoint_max=0,
        )
        worker._stub = _Master(master)

        for var in worker._model.trainable_variables:
            master.set_model_var(var.name, var.numpy())

        try:
            worker.run()
            res = True
        except Exception as ex:
            print(ex)
            res = False

        self.assertTrue(res)
        req = elasticdl_pb2.GetTaskRequest()
        req.worker_id = 1
        task = master.GetTask(req, None)
        # No more task.
        self.assertTrue(not task.shard_file_name)

    def test_distributed_train_bytes(self):
        self.distributed_train_and_evaluate("bytes", training=True)

    def test_distributed_evaluate_bytes(self):
        self.distributed_train_and_evaluate("bytes", training=False)

    def test_distributed_train_tf_example(self):
        self.distributed_train_and_evaluate("tf_example", training=True)

    def test_distributed_evaluate_tf_example(self):
        self.distributed_train_and_evaluate("tf_example", training=False)


if __name__ == "__main__":
    unittest.main()
