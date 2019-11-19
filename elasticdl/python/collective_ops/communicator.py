# TODO: This is dummy for now until the real implementation
# has been open sourced


class CollectiveCommunicator(object):
    def __init__(self):
        pass

    def allreduce(self, data, op="MEAN"):
        return True, 1

    def broadcast(self, from_worker_ip):
        return True, {"param1": 1}

    def barrier(self):
        return True
