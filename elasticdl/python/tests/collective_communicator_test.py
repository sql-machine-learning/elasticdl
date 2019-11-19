import unittest

from elasticdl.python.collective_ops.communicator import CollectiveCommunicator
from elasticdl.python.common.constants import CollectiveCommunicatorStatus


class CollectiveCommunicatorTest(unittest.TestCase):
    def test_collective_communicator(self):
        communicator = CollectiveCommunicator()
        self.assertEqual(
            communicator.allreduce([1]),
            (CollectiveCommunicatorStatus.SUCCEEDED, [1]),
        )
        self.assertEqual(
            communicator.broadcast("worker_0_ip"),
            (CollectiveCommunicatorStatus.SUCCEEDED, {"param1": 1}),
        )
        self.assertEqual(
            communicator.barrier(), CollectiveCommunicatorStatus.SUCCEEDED
        )


if __name__ == "__main__":
    unittest.main()
