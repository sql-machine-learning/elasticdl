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

import os
import time
import traceback

from elasticai_api.common.constants import WorkerEnv
from elasticdl.python.allreduce.base_controller import (
    DEFAULT_MAX_ALLREDUCE_RETRY_NUM,
    AllReduceController,
)
from elasticdl.python.common.grpc_utils import build_channel
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.worker.data_shard_service import DataShardService
from elasticdl.python.worker.master_client import MasterClient

try:
    import horovod.torch as hvd
    from horovod.common.exceptions import HorovodInternalError
    from horovod.torch.functions import (
        broadcast_optimizer_state,
        broadcast_parameters,
    )

except ImportError:
    hvd = None


def create_elastic_controller(batch_size):
    """Create an elastic AllReduce controller with data shard service.
    Users can use the `controller.data_shard_service` to get data
    shards like:
    ```python
    while True:
        shard = controller.data_shard_service.fetch_shard()
        for i in range(shard.start, shard.end):
            yield i
    ```

    Users also can use the controller to do an elastic training.

    ```python
    model = ...
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    optimizer = hvd.DistributedOptimizer(optimizer)

    controller.set_broadcast_model(model)
    ontroller.set_broadcast_optimizer(optimizer)
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        # Use the elastic function to wrap the training function with a batch.
        elastic_train_one_batch = allreduce_controller.elastic_run(
            train_one_batch
        )

    def train_one_batch(model, optimizer, data, target):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        return loss
    ```
    """
    master_addr = os.getenv("MASTER_ADDR", "localhost:12345")
    worker_id = int(os.getenv("WORKER_ID", 0))

    master_client = MasterClient(build_channel(master_addr), worker_id)
    data_shard_service = DataShardService(batch_size, master_client)

    controller = PyTorchAllReduceController(master_client, data_shard_service)
    controller.init_horovod_locally()
    return controller


class PyTorchAllReduceController(AllReduceController):
    def __init__(self, master_client, data_shard_service):
        super(PyTorchAllReduceController, self).__init__(
            master_client, data_shard_service
        )
        self._model = None
        self._optimizer = None
        self.backward_passes_per_step = 1
        # ElasticDL master should set the number of workers into envs.
        self.batch_num_per_step = int(os.getenv(WorkerEnv.WORKER_NUM, 1))

    def set_broadcast_model(self, model):
        self._model = model

    def set_broadcast_optimizer(self, optimizer):
        self._optimizer = optimizer

    def broadcast(self):
        broadcast_parameters(self._model.state_dict(), root_rank=0)
        broadcast_optimizer_state(self._optimizer, root_rank=0)

    def train_one_batch_with_retries(self, func, *args, **kwargs):
        self.reset_backward_passes_per_step()
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
        self._optimizer.zero_grad()
        self._rendezvous_manager.init_horovod_if_needed()

    def reset_backward_passes_per_step(self):
        world_size = hvd.size()
        rank = hvd.rank()
        self.backward_passes_per_step = int(
            self.batch_num_per_step / world_size
        )
        if rank < self.batch_num_per_step % world_size:
            self.backward_passes_per_step += 1
        if (
            self.backward_passes_per_step
            != self._optimizer.backward_passes_per_step
        ):
            self._optimizer.backward_passes_per_step = (
                self.backward_passes_per_step
            )
            logger.info(
                "Backward passes = {}".format(
                    self._optimizer.backward_passes_per_step
                )
            )
