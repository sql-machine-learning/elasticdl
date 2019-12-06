import unittest

from elasticdl.python.collective_ops.communicator import CollectiveCommunicator
from elasticdl.python.common.constants import CollectiveCommunicatorStatus


class CollectiveCommunicatorTest(unittest.TestCase):
    def test_collective_communicator(self):
        communicator = CollectiveCommunicator()
        data = [1]
        self.assertEqual(
            communicator.allreduce(data),
            (CollectiveCommunicatorStatus.SUCCEEDED, data),
        )
        self.assertEqual(
            communicator.allreduce(None),
            (CollectiveCommunicatorStatus.FAILED, None),
        )
        data = {"param1": 1}
        self.assertEqual(
            communicator.broadcast(data, "worker_0_ip"),
            (CollectiveCommunicatorStatus.SUCCEEDED, data),
        )
        self.assertEqual(
            communicator.broadcast(None, "worker_0_ip"),
            (CollectiveCommunicatorStatus.SUCCEEDED, None),
        )
        self.assertEqual(
            communicator.barrier(), CollectiveCommunicatorStatus.SUCCEEDED
        )


if __name__ == "__main__":
    unittest.main()
