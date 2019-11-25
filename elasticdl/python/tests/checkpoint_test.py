import os
import unittest

from elasticdl.python.common.model_utils import (
    get_module_file_path,
    load_module,
)
from elasticdl.python.master.checkpoint_service import CheckpointService

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


if __name__ == "__main__":
    unittest.main()
