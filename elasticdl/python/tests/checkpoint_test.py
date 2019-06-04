import random
import os
import mock
import grpc
import tempfile
import unittest
import numpy as np
import recordio

import tensorflow as tf

tf.enable_eager_execution()

from contextlib import closing
from elasticdl.python.elasticdl.master.servicer import MasterServicer
from elasticdl.python.elasticdl.worker.worker import Worker
from elasticdl.python.elasticdl.common.model_helper import load_user_model
from elasticdl.python.elasticdl.master.task_queue import _TaskQueue
from elasticdl.python.elasticdl.common.model_helper import save_checkpoint_to_file, load_from_checkpoint_file
from elasticdl.proto import elasticdl_pb2
from elasticdl.python.data.codec import BytesCodec


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


class CheckpointTest(unittest.TestCase):
    def testSaveLoadCheckpoint(self):
        master = MasterServicer(
            2, 3, None, None, None,
            init_var=[],
            checkpoint_dir="",
            checkpoint_steps=0,
            keep_checkpoint_max=0
        )

        model_inst = m.model
        for variable in model_inst.trainable_variables:
            master.set_model_var(variable.name, variable.numpy())

        req = elasticdl_pb2.GetModelRequest()
        req.min_version = 0
        model = master.GetModel(req, None)

        tmp_file = tempfile.NamedTemporaryFile()
        save_checkpoint_to_file(model, tmp_file.name)

        pb_model = load_from_checkpoint_file(tmp_file.name)

        self.assertEqual(model.version, pb_model.version)
        for k in model.param:
            self.assertEqual(model.param[k], pb_model.param[k])

    def testCheckpintArguments(self):
        """
        Run Worker.distributed_train with a local master.
        grpc calls are mocked by local master call.
        """

        def mock_GetTask(req):
            return master.GetTask(req, None)

        def mock_GetModel(req):
            return master.GetModel(req, None)

        def mock_ReportGradient(req):
            return master.ReportGradient(req, None)

        def mock_ReportTaskResult(req):
            return master.ReportTaskResult(req, None)

        codec_type = "bytes"
        channel = grpc.insecure_channel("localhost:9999")
        worker = Worker(
            1,
            _module_file,
            channel=channel,
            codec_type=codec_type,
        )

        # save checkpoint file every 2 steps
        # keep at most 5 recent checkpoint files
        checkpoint_dir = tempfile.mkdtemp()
        checkpoint_steps = 2
        keep_checkpoint_max = 5

        filename = create_recordio_file(128, codec_type)
        task_q = _TaskQueue({filename: 128}, record_per_task=64, num_epoch=1)
        master = MasterServicer(2,
                                2,
                                worker._opt_fn(),
                                task_q,
                                None,
                                init_var=[],
                                checkpoint_dir=checkpoint_dir,
                                checkpoint_steps=checkpoint_steps,
                                keep_checkpoint_max=keep_checkpoint_max
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
            worker._stub, "ReportTaskResult", mock_ReportTaskResult
        ):
            try:
                worker.distributed_train()
                res = True
            except Exception as ex:
                print(ex)
                res = False

        self.assertTrue(res)
        checkpoint_files = sorted(os.listdir(checkpoint_dir))
        self.assertEqual(checkpoint_files,
                         ['model_v24.chkpt', 'model_v26.chkpt', 'model_v28.chkpt', 'model_v30.chkpt', 'model_v32.chkpt'])


if __name__ == "__main__":
    unittest.main()
