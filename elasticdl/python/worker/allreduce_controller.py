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
from abc import abstractmethod
from functools import wraps

from tensorflow.python.framework.errors_impl import UnknownError

from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.worker.allreduce_trainer import (
    DEFAULT_MAX_ALLREDUCE_RETRY_NUM,
    DEFAULT_STEPS_TO_CHECK_RENDEZVOUS,
    RendevousManager,
)

try:
    from horovod.tensorflow.functions import broadcast_variables
    import horovod.tensorflow as hvd
except ImportError:
    hvd = None


class AllReduceController(object):
    """The controller initializes Horovod and calls the function with forward
    and backward computation using a mini-batch of data. If Horovod raise an
    exception about AllReduce, Allgather and Broadcast, the controller will
    catch the exception and re-initialize Horovod. Then, it will broadcast
    the variables and retry to call those functions.
    """

    def __init__(self, master_client, master_addr):
        if not hvd:
            raise RuntimeError("Horovod is not installed for AllReduce")

        self._rendezvous_manager = RendevousManager(master_client, master_addr)
        self._step = 0
        self._first_call = True
        self._need_broadcast = True

    def elastic_run(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self._init_variables_before_first_calling(func, *args, **kwargs)
            self._init_horovod_periodically()
            return self._try_to_call_func(func, *args, **kwargs)

        return wrapper

    def _init_variables_before_first_calling(self, func, *args, **kwargs):
        if self._first_call:
            hvd.init()
            func(*args, **kwargs)
            self._first_call = False

    def _init_horovod_periodically(self):
        """Check whether to initialize Horovod periodically in case
        that new workers join the job.
        """
        if self._step % DEFAULT_STEPS_TO_CHECK_RENDEZVOUS == 0:
            self._rendezvous_manager.init_horovod_if_needed()

    def _broadcast_if_needed(self):
        if self._rendezvous_manager.need_broadcast:
            logger.info("Broadcast models")
            self.broadcast()
            self._rendezvous_manager.need_broadcast = False

    def _try_to_call_func(self, func, *args, **kwargs):
        for _ in range(DEFAULT_MAX_ALLREDUCE_RETRY_NUM + 1):
            try:
                self._broadcast_if_needed()
                result = func(*args, **kwargs)
                self._step += 1
                return result
            except UnknownError as e:
                logger.warning(
                    "Failed to perform allreduce operation on "
                    "the gradients. Retrying..."
                )
                # Those error message show that the communication
                # to merge gradient fails and we can rebuild the
                # communication.
                if (
                    "HorovodAllreduce" in e.message
                    or "HorovodAllgather" in e.message
                    or "HorovodBroadcast" in e.message
                ):
                    time.sleep(3)
                    self._rendezvous_manager.init_horovod_if_needed()

    @abstractmethod
    def broadcast(self):
        pass


class TensorFlowV2AllreduceController(AllReduceController):
    """The controller is responsible for elastic training of
    TensorFlow eager execution using AllReduce.
    """

    def __init__(self, master_client, master_addr):
        super(TensorFlowV2AllreduceController, self).__init__(
            master_client, master_addr
        )
        self._model = None
        self._optimizer = None

    def set_broadcast_model(self, model):
        self._model = model

    def set_broadcast_optimizer(self, optimizer):
        self._optimizer = optimizer

    def broadcast(self):
        broadcast_variables(self._model.variables, root_rank=0)
        broadcast_variables(self._optimizer.variables(), root_rank=0)
