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

import time
import traceback

from elasticdl.python.allreduce.base_controller import (
    DEFAULT_MAX_ALLREDUCE_RETRY_NUM,
    AllReduceController,
)
from elasticdl.python.common.log_utils import default_logger as logger

try:
    import horovod.torch as hvd
    from horovod.common.exceptions import HorovodInternalError

except ImportError:
    hvd = None


class PyTorchAllReduceController(AllReduceController):
    def __init__(self, master_client, master_addr, data_shard_service):
        super(PyTorchAllReduceController, self).__init__(
            master_client, master_addr, data_shard_service
        )
        self._model = None
        self._optimizer = None

    def set_broadcast_model(self, model):
        self._model = model

    def set_broadcast_optimizer(self, optimizer):
        self._optimizer = optimizer

    def broadcast(self):
        from horovod.torch.functions import (
            broadcast_optimizer_state,
            broadcast_parameters,
        )

        broadcast_parameters(self._model.state_dict(), root_rank=0)
        broadcast_optimizer_state(self._optimizer, root_rank=0)

    def train_one_batch_with_retries(self, func, *args, **kwargs):
        for _ in range(DEFAULT_MAX_ALLREDUCE_RETRY_NUM):
            try:
                self._broadcast_if_needed()
                result = func(*args, **kwargs)
                break
            except HorovodInternalError:
                logger.warning(
                    "Failed to perform allreduce operation on "
                    "the gradients. Retrying..."
                )
                # Those error message show that the communication
                # to merge gradient fails and we can rebuild the
                # communication.
                self.restore()
            except RuntimeError:
                traceback.print_exc()
                self.restore()
        self.data_shard_service.report_batch_done()
        return result

    def restore(self):
        time.sleep(3)
        # Call `load_state_dict` to reset the state of Horovod optimizer
        self._optimizer.load_state_dict(self._optimizer.state_dict())
        self._rendezvous_manager.init_horovod_if_needed()
