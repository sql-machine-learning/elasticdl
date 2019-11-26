import os
import tempfile
import unittest

import numpy as np

from elasticdl.python.common.model_utils import (
    get_module_file_path,
    load_module,
)
from elasticdl.python.common.save_utils import CheckpointSaver
from elasticdl.python.ps.parameters import Parameters

_model_zoo_path = os.path.dirname(os.path.realpath(__file__))
_model_file = get_module_file_path(_model_zoo_path, "test_module.custom_model")
m = load_module(_model_file).__dict__


class SaveUtilsTest(unittest.TestCase):
    def testNeedToCheckpoint(self):
        checkpointer = CheckpointSaver("", 0, 5, False)
        self.assertFalse(checkpointer.is_enabled())
        checkpointer._steps = 3
        self.assertTrue(checkpointer.is_enabled())

        self.assertFalse(checkpointer.need_to_checkpoint(1))
        self.assertFalse(checkpointer.need_to_checkpoint(2))
        self.assertTrue(checkpointer.need_to_checkpoint(3))
        self.assertFalse(checkpointer.need_to_checkpoint(4))
        self.assertFalse(checkpointer.need_to_checkpoint(5))
        self.assertTrue(checkpointer.need_to_checkpoint(6))

    def testGetCheckpointPath(self):
        ckpt_dir = "test/checkpoint_dir"
        checkpoint_saver = CheckpointSaver(ckpt_dir, 3, 5, False)
        checkpint_path = checkpoint_saver._get_checkpoint_file(100)
        self.assertEqual(
            checkpint_path,
            "test/checkpoint_dir/version-100/variables-0-of-1.ckpt",
        )

    def testSaveLoadCheckpoint(self):
        init_var = m["custom_model"]().trainable_variables
        with tempfile.TemporaryDirectory() as tempdir:
            ckpt_dir = os.path.join(tempdir, "testSaveLoadCheckpoint")
            os.makedirs(ckpt_dir)
            checkpoint_saver = CheckpointSaver(ckpt_dir, 3, 5, False)
            self.assertTrue(checkpoint_saver.is_enabled())
            params = Parameters()

            for var in init_var:
                params.non_embedding_params[var.name] = var
            pb_model = params.to_model_pb()

            checkpoint_saver.save(0, pb_model, False)

            ckpt_version_dir = os.path.join(ckpt_dir, "version-0")
            restore_params = CheckpointSaver.restore_params_from_checkpoint(
                ckpt_version_dir, 0, 1
            )
            self.assertEqual(restore_params.version, params.version)
            for var_name in params.non_embedding_params:
                self.assertTrue(
                    np.array_equal(
                        params.non_embedding_params[var_name].numpy(),
                        restore_params.non_embedding_params[var_name].numpy(),
                    )
                )


if __name__ == "__main__":
    unittest.main()
