# Copyright 2020 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import socket

from elasticdl.python.common.constants import CollectiveCommunicatorStatus
from elasticdl.python.common.log_utils import default_logger as logger

try:
    from ftlib import BasicFTLib
    from ftlib.commlib.commlib_status import CommLibStatus
    from ftlib.ftlib_status import FTCollectiveStatus

    _FTLIB_INSTALLED = True
except ImportError:
    BasicFTLib = object
    FTCollectiveStatus = object
    _FTLIB_INSTALLED = False


_SUPPORTED_ALLREDUCE_OPS = ["MEAN"]
_FTLIB_UNINSTALLED_DEFAULT_STATUS_MESSAGE = (
    "FTLib is not installed. Default to succeeded for testing purposes"
)


class CollectiveCommunicator(object):
    def __init__(self, service_name=None):
        if _FTLIB_INSTALLED:
            peer_list = list(self._get_peer_set(service_name))
            self._ftlib = BasicFTLib(
                consensus="gossip",
                commlib="pytorch",
                consensus_init_kwargs={
                    "known_addr_list": peer_list,
                    "custom_bind_addr": socket.gethostbyname(
                        socket.gethostname()
                    ),
                },
            )
            while peer_list and not self._ftlib.consensus_joined():
                logger.warning("Retry building consensus...")
                self._ftlib.manual_join(
                    known_addr_list=list(self._get_peer_set(service_name))
                )
        else:
            logger.warning(
                "FTLib is not installed. The CollectiveCommunicator "
                "may not work as expected"
            )
            self._ftlib = None

    def tf_allreduce(self, grads, op="MEAN"):
        if grads is None:
            logger.error("Grads is required for tf_allreduce operation")
            return CollectiveCommunicatorStatus.FAILED, grads
        # convert tf.Tensor to numpy
        numpy_data = [g.numpy() for g in grads]
        return self.allreduce(numpy_data, op)

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
            status, res = self._ftlib.wait_gradients_ready(params=data)
            if (
                status == FTCollectiveStatus.SUCCESS
                and res == CommLibStatus.SUCCESS
                or status == FTCollectiveStatus.NO_NEED
            ):
                return CollectiveCommunicatorStatus.SUCCEEDED, data
            else:
                return CollectiveCommunicatorStatus.FAILED, data
        else:
            logger.warning(_FTLIB_UNINSTALLED_DEFAULT_STATUS_MESSAGE)
            return CollectiveCommunicatorStatus.SUCCEEDED, data

    def tf_broadcast(self, params, src_rank):
        for p in params:
            data = p.numpy()
            status, data = self.broadcast(p.numpy(), src_rank)
            if status == CollectiveCommunicatorStatus.SUCCEEDED:
                p.assign(data)
            else:
                return status
        return CollectiveCommunicatorStatus.SUCCEEDED

    def broadcast(self, data, src_rank):
        if self._ftlib is not None:
            status, _ = self._ftlib.broadcast(data, src_rank)
            if status == FTCollectiveStatus.SUCCESS:
                return CollectiveCommunicatorStatus.SUCCEEDED, data
            else:
                return CollectiveCommunicatorStatus.FAILED, data
        else:
            logger.warning(_FTLIB_UNINSTALLED_DEFAULT_STATUS_MESSAGE)
            return CollectiveCommunicatorStatus.SUCCEEDED, data

    def barrier(self):
        if self._ftlib is not None:
            status, _ = self._ftlib.barrier()
            if status == FTCollectiveStatus.SUCCESS:
                return CollectiveCommunicatorStatus.SUCCEEDED
            else:
                return CollectiveCommunicatorStatus.FAILED
        else:
            logger.warning(_FTLIB_UNINSTALLED_DEFAULT_STATUS_MESSAGE)
            return CollectiveCommunicatorStatus.SUCCEEDED

    def is_initialized(self):
        """This will be `False` under three occasions:
           * New workers report joining in
           * Collective-communication operations fail or time out
           * Liveness probe fails for existing workers
        """
        if self._ftlib is not None:
            return self._ftlib.initialized
        else:
            return True

    def _get_peer_set(self, svc_name):
        if svc_name is None:
            return None
        my_ip = socket.gethostbyname(socket.gethostname())
        temp_set = socket.getaddrinfo(svc_name, 0, proto=socket.IPPROTO_TCP)
        peer_set = {peer[-1][0] for peer in temp_set if peer[-1][0] != my_ip}
        return peer_set
