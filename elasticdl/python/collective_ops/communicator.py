# TODO: This is dummy for now until the real implementation
# has been open sourced

from elasticdl.python.common.constants import CollectiveCommunicatorStatus


class CollectiveCommunicator(object):
    def __init__(self):
        pass

    def allreduce(self, data, op="MEAN"):
        return CollectiveCommunicatorStatus.SUCCEEDED, data

    def broadcast(self, from_worker_ip):
        return CollectiveCommunicatorStatus.SUCCEEDED, {"param1": 1}

    def barrier(self):
        return CollectiveCommunicatorStatus.SUCCEEDED
