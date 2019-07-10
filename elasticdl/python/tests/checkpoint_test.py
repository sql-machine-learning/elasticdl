import os
import tempfile
import unittest
from contextlib import closing

import numpy as np
import recordio
import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.model_helper import load_module
from elasticdl.python.master.checkpoint_service import CheckpointService
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.master.task_dispatcher import _TaskDispatcher
from elasticdl.python.tests.in_process_master import InProcessMaster
from elasticdl.python.worker.worker import Worker

_module_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_module.py"
)

m = load_module(_module_file)


def create_recordio_file(size):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with closing(recordio.Writer(temp_file.name)) as f:
        for _ in range(size):
            x = np.random.rand(1).astype(np.float32)
            y = 2 * x + 1
            example_dict = {
                "x": tf.train.Feature(float_list=tf.train.FloatList(value=x)),
                "y": tf.train.Feature(float_list=tf.train.FloatList(value=y)),
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=example_dict)
            )
            f.write(example.SerializeToString())
    return temp_file.name


class CheckpointTest(unittest.TestCase):
    def testNeedToCheckpoint(self):
        checkpointer = CheckpointService("", 0, 5, False)
        self.assertFalse(checkpointer.is_enabled())
        checkpointer._steps = 3
        self.assertTrue(checkpointer.is_enabled())

        self.assertFalse(checkpointer.need_to_checkpoint(1))
        self.assertFalse(checkpointer.need_to_checkpoint(2))
        self.assertTrue(checkpointer.need_to_checkpoint(3))
        self.assertFalse(checkpointer.need_to_checkpoint(4))
        self.assertFalse(checkpointer.need_to_checkpoint(5))
        self.assertTrue(checkpointer.need_to_checkpoint(6))

    def testSaveLoadCheckpoint(self):
        init_var = m.custom_model().trainable_variables
        with tempfile.TemporaryDirectory() as tempdir:
            chkp_dir = os.path.join(tempdir, "testSaveLoadCheckpoint")
            os.makedirs(chkp_dir)
            checkpointer = CheckpointService(chkp_dir, 3, 5, False)
            self.assertTrue(checkpointer.is_enabled())

            master = MasterServicer(
                2,
                3,
                None,
                None,
                init_var=init_var,
                checkpoint_filename_for_init="",
                checkpoint_service=checkpointer,
                evaluation_service=None,
            )

            req = elasticdl_pb2.GetModelRequest()
            req.method = elasticdl_pb2.MINIMUM
            req.version = 0
            model = master.GetModel(req, None)
            checkpointer.save(0, model, False)
            loaded_model = checkpointer.get_checkpoint_model(0)
            self.assertEqual(model.version, loaded_model.version)
            for k in model.param:
                self.assertEqual(model.param[k], loaded_model.param[k])

    def testMaxCheckpointVersions(self):
        with tempfile.TemporaryDirectory() as tempdir:
            chkp_dir = os.path.join(tempdir, "testMaxCheckpointVersions")
            os.makedirs(chkp_dir)
            # Save checkpoints every 2 steps, and keep 5 checkpoints at most
            checkpointer = CheckpointService(chkp_dir, 2, 5, False)
            self.assertTrue(checkpointer.is_enabled())

            # Launch the training
            worker = Worker(1, _module_file, channel=None)
            filename = create_recordio_file(128)
            task_d = _TaskDispatcher(
                {filename: 128}, {}, {}, records_per_task=64, num_epochs=1
            )
            master = MasterServicer(
                2,
                2,
                worker._opt_fn(),
                task_d,
                init_var=worker._model.trainable_variables,
                checkpoint_filename_for_init="",
                checkpoint_service=checkpointer,
                evaluation_service=None,
            )

            worker._stub = InProcessMaster(master)
            worker.run()

            # We should have 5 checkpoints when the training finishes
            checkpoint_files = sorted(os.listdir(checkpointer._directory))
            self.assertEqual(
                checkpoint_files,
                [
                    "model_v24.chkpt",
                    "model_v26.chkpt",
                    "model_v28.chkpt",
                    "model_v30.chkpt",
                    "model_v32.chkpt",
                ],
            )
            # Latest version should be 32
            self.assertEqual(32, checkpointer.get_latest_checkpoint_version())
            # Check all checkpoints
            for version in [24, 26, 28, 30, 32]:
                model = checkpointer.get_checkpoint_model(version)
                self.assertEqual(version, model.version)
            # Checkpoint not found
            self.assertRaisesRegex(
                RuntimeError,
                "Failed to read model checkpoint from file",
                checkpointer.get_checkpoint_model,
                100,
            )

    def testInitFromCheckpoint(self):
        init_var = m.custom_model().trainable_variables
        with tempfile.TemporaryDirectory() as tempdir:
            chkp_dir = os.path.join(tempdir, "testInitFromCheckpoint")
            os.makedirs(chkp_dir)
            master = MasterServicer(
                2,
                3,
                None,
                None,
                init_var=init_var,
                checkpoint_filename_for_init="",
                checkpoint_service=CheckpointService(chkp_dir, 2, 3, False),
                evaluation_service=None,
            )
            req = elasticdl_pb2.GetModelRequest()
            req.method = elasticdl_pb2.MINIMUM
            req.version = 0
            model = master.GetModel(req, None)
            master._checkpoint_service.save(master._version, model, False)

            chkp_file = master._checkpoint_service.get_checkpoint_path(
                master._version
            )
            # Create variables from init_var, get init value from checkpoint.
            master2 = MasterServicer(
                2,
                3,
                None,
                None,
                init_var=init_var,
                checkpoint_filename_for_init=chkp_file,
                checkpoint_service=CheckpointService("", 0, 0, False),
                evaluation_service=None,
            )
            model2 = master2.GetModel(req, None)
            self.assertEqual(model, model2)
            # Create variables from checkpoint.
            master3 = MasterServicer(
                2,
                3,
                None,
                None,
                init_var=[],
                checkpoint_filename_for_init=chkp_file,
                checkpoint_service=CheckpointService("", 0, 0, False),
                evaluation_service=None,
            )
            model3 = master3.GetModel(req, None)
            self.assertEqual(model, model3)


if __name__ == "__main__":
    unittest.main()
