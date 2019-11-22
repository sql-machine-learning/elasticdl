# TODO: This is dummy for now until the real implementation
# has been open sourced

from elasticdl.python.common.constants import CollectiveCommunicatorStatus


class CollectiveCommunicator(object):
    def __init__(self):
        pass

    def allreduce(self, data, op="MEAN"):
        if data is None:
            return CollectiveCommunicatorStatus.FAILED, data
        return CollectiveCommunicatorStatus.SUCCEEDED, data

    def broadcast(self, data, root_ip):
        if data is None:
            return CollectiveCommunicatorStatus.FAILED, data
        return CollectiveCommunicatorStatus.SUCCEEDED, data

    def barrier(self):
        return CollectiveCommunicatorStatus.SUCCEEDED

    def has_new_worker_joining(self):
        return True
