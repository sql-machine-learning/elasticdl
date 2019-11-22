import os
import tempfile
import unittest

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.model_utils import (
    get_module_file_path,
    load_module,
)
from elasticdl.python.master.checkpoint_service import CheckpointService
from elasticdl.python.master.servicer import MasterServicer

_model_zoo_path = os.path.dirname(os.path.realpath(__file__))
_model_file = get_module_file_path(_model_zoo_path, "test_module.custom_model")
m = load_module(_model_file).__dict__


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
        init_var = m["custom_model"]().trainable_variables
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
            for var, loaded_var in zip(model.param, loaded_model.param):
                self.assertEqual(var, loaded_var)

    def testInitFromCheckpoint(self):
        init_var = m["custom_model"]().trainable_variables
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
