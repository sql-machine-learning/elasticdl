import socket

from elasticdl.python.common.constants import CollectiveCommunicatorStatus
from elasticdl.python.common.log_utils import default_logger as logger

try:
    from ftlib import BasicFTLib
    from ftlib.ftlib_status import FTAllReduceStatus

    _FTLIB_INSTALLED = True
except ImportError:
    BasicFTLib = object
    FTAllReduceStatus = object
    _FTLIB_INSTALLED = False


_SUPPORTED_ALLREDUCE_OPS = ["MEAN"]


class CollectiveCommunicator(object):
    def __init__(self, svc_name=None):
        if _FTLIB_INSTALLED:
            self._ftlib = BasicFTLib(
                consensus="gossip",
                commlib="pytorch",
                consensus_init_kwargs={
                    "known_addr_list": list(self._get_peer_set(svc_name))
                },
            )
        else:
            logger.warning(
                "FTLib is not installed. The CollectiveCommunicator "
                "may not work as expected"
            )
            self._ftlib = None

    def allreduce(self, data, op="MEAN"):
        if data is None:
            logger.error("Data is required for allreduce operation")
            return CollectiveCommunicatorStatus.FAILED, data
        if op not in _SUPPORTED_ALLREDUCE_OPS:
            logger.error(
                "%s is not in list of supported allreduce operations: %s"
                % (op, _SUPPORTED_ALLREDUCE_OPS)
            )
            return CollectiveCommunicatorStatus.FAILED, data
        if self._ftlib is not None:
            res = self._ftlib.allreduce_average(data)
            if res == FTAllReduceStatus.SUCCESS:
                return CollectiveCommunicatorStatus.SUCCEEDED, data
            else:
                return CollectiveCommunicatorStatus.FAILED, data
        else:
            logger.warning(
                "FTLib is not installed. "
                "Default to succeeded for testing purposes"
            )
            return CollectiveCommunicatorStatus.SUCCEEDED, data

    def broadcast(self, data, root_ip):
        if self._ftlib is not None:
            res = self._ftlib.broadcast(data, root_ip)
            if res == FTAllReduceStatus.SUCCESS:
                return CollectiveCommunicatorStatus.SUCCEEDED, data
            else:
                return CollectiveCommunicatorStatus.FAILED, data
        else:
            logger.warning(
                "FTLib is not installed. "
                "Default to succeeded for testing purposes"
            )
            return CollectiveCommunicatorStatus.SUCCEEDED, data

    def barrier(self):
        return CollectiveCommunicatorStatus.SUCCEEDED

    def has_new_worker_joining(self):
        return True

    def _get_peer_set(self, svc_name):
        if svc_name is None:
            return None
        my_ip = socket.gethostbyname(socket.gethostname())
        temp_set = socket.getaddrinfo(svc_name, 0, proto=socket.IPPROTO_TCP)
        peer_set = {peer[-1][0] for peer in temp_set if peer[-1][0] != my_ip}
        return peer_set
