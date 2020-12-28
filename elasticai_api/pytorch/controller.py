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

from elasticai_api.common.base_controller import (
    DEFAULT_MAX_ALLREDUCE_RETRY_NUM,
    RETRY_ALLREDUCE_INTERVAL_SECS,
    AllReduceController,
)
from elasticai_api.common.constants import WorkerEnv
from elasticai_api.common.data_shard_service import RecordIndexService
from elasticai_api.common.master_client import build_master_client
from elasticai_api.util.log_utils import default_logger as logger

try:
    import horovod.torch as hvd
    from horovod.common.exceptions import HorovodInternalError
    from horovod.torch.functions import (
        broadcast_optimizer_state,
        broadcast_parameters,
        broadcast_object,
    )

except ImportError:
    hvd = None


def create_elastic_controller(
    batch_size, num_epochs=None, dataset_size=None, shuffle=False
):
    """Create an elastic AllReduce controller with record index service.
    Users can use the `controller.data_shard_service` to get data
    shards like:
    ```python
    index = controller.data_shard_service.fetch_record_index()
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

    Args:
        batch_size: The batch size of a single worker.
        num_epochs: The number of epochs.
        dataset_size: The total size of dataset.
    """
    master_client = build_master_client()
    record_index_service = RecordIndexService(
        master_client=master_client,
        batch_size=batch_size,
        num_epochs=num_epochs,
        dataset_size=dataset_size,
        shuffle=shuffle,
    )

    controller = PyTorchAllReduceController(
        master_client, record_index_service
    )
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
        self.global_batch_num_per_step = int(
            os.getenv(WorkerEnv.WORKER_NUM, 1)
        )
        self.global_completed_batch_num = 0

    def set_broadcast_model(self, model):
        self._model = model

    def set_broadcast_optimizer(self, optimizer):
        self._optimizer = optimizer

    def broadcast(self):
        broadcast_parameters(self._model.state_dict(), root_rank=0)
        broadcast_optimizer_state(self._optimizer, root_rank=0)
        self.global_completed_batch_num = broadcast_object(
            self.global_completed_batch_num, name="GlobalCompletedBatchNum"
        )

    def train_one_batch_with_retries(self, func, *args, **kwargs):
        self.reset_backward_passes_per_step()
        allreduce_success = False
        for _ in range(DEFAULT_MAX_ALLREDUCE_RETRY_NUM):
            try:
                self._broadcast_if_needed()
                result = func(*args, **kwargs)
                allreduce_success = True
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
        if not allreduce_success:
            raise RuntimeError("Failed to perform allreduce.")
        self._update_completed_minibatches()
        return result

    def restore(self):
        time.sleep(RETRY_ALLREDUCE_INTERVAL_SECS)
        # Call `load_state_dict` to reset the state of Horovod optimizer
        self._optimizer.load_state_dict(self._optimizer.state_dict())
        self._optimizer.zero_grad()
        self._rendezvous_manager.init_horovod_if_needed()

    def _update_completed_minibatches(self):
        if (
            hasattr(self._optimizer, "fixed_global_batch_size")
            and self._optimizer.fixed_global_batch_size
        ):
            if self._optimizer.update_gradients:
                self.global_completed_batch_num += (
                    self.global_batch_num_per_step
                )
        else:
            self.global_completed_batch_num += hvd.size()

    def reset_backward_passes_per_step(self):
        # Only reset backward_passes_per_step when using the optimizer
        # with fixed_global_batch_size
        if (
            hasattr(self._optimizer, "fixed_global_batch_size")
            and self._optimizer.fixed_global_batch_size
        ):
            world_size = hvd.size()
            rank = hvd.rank()
            self.backward_passes_per_step = int(
                self.global_batch_num_per_step / world_size
            )
            if rank < self.global_batch_num_per_step % world_size:
                self.backward_passes_per_step += 1
            if (
                self.backward_passes_per_step
                != self._optimizer.backward_passes_per_step
            ):
                self._optimizer.set_backward_passes_per_step(
                    self.backward_passes_per_step
                )
                logger.info(
                    "Backward passes per step = {}".format(
                        self._optimizer.backward_passes_per_step
                    )
                )
