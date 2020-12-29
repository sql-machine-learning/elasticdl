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
import socket
import time
from abc import abstractmethod
from contextlib import contextmanager
from functools import wraps

from elasticai_api.common.constants import (
    HorovodEnv,
    TrainingLoopStatus,
    WorkerEnv,
)
from elasticai_api.util.log_utils import default_logger as logger

try:
    if os.getenv("USE_TORCH", None):
        import horovod.torch as hvd
    else:
        import horovod.tensorflow as hvd

except ImportError:
    hvd = None


# The default maximum number of retries for allreduce operation
# if allreduce-based distributed training strategy is used.
DEFAULT_MAX_ALLREDUCE_RETRY_NUM = 5
# The default timeout is 30s in Horovod.
# https://github.com/horovod/horovod/blob/2fdea15bc6317848944c72cf8dd0aaa98b2e1a2a/horovod/common/gloo/gloo_context.cc#L59
DEFAULT_SECS_TO_CHECK_RENDEZVOUS = min(
    60, int(os.getenv(HorovodEnv.GLOO_TIMEOUT_SECONDS, 30))
)
RETRY_ALLREDUCE_INTERVAL_SECS = 30


class RendevousManager(object):
    def __init__(self, master_client):
        self.need_broadcast = True
        self._master_client = master_client
        self._rendezvous_id = None

    def init_horovod_if_needed(self):
        self._set_horovod_env()
        for _ in range(DEFAULT_MAX_ALLREDUCE_RETRY_NUM):
            rank_response = self._master_client.get_comm_rank()
            if rank_response.rank_id < 0:
                logger.warning(
                    "The master has not added the worker host into "
                    "rendezvous yet. Retrying to get rank"
                )
                time.sleep(RETRY_ALLREDUCE_INTERVAL_SECS)
            else:
                break

        # If the rendezvous from master is unequal to self._rendezvous_id,
        # the worker should rebuild the communication because the master
        # has updated the communication group.
        if rank_response.rendezvous_id != self._rendezvous_id:
            logger.info(
                "Initialize Horovod with rank = {} and size = {}".format(
                    rank_response.rank_id, rank_response.world_size
                )
            )
            os.environ[HorovodEnv.RENDEZVOUS_PORT] = str(
                rank_response.rendezvous_port
            )
            os.environ[HorovodEnv.RANK] = str(rank_response.rank_id)
            os.environ[HorovodEnv.SIZE] = str(rank_response.world_size)
            # Not using Horovod elastic feature in init, but need it for
            # allreduce to call allreduce op when size=1.
            os.environ[HorovodEnv.ELASTIC] = str(0)
            hvd.shutdown()
            hvd.init()
            os.environ[HorovodEnv.ELASTIC] = str(1)
            self._rendezvous_id = rank_response.rendezvous_id
            self.need_broadcast = True

    def _set_horovod_env(self):
        master_addr_port = os.getenv(WorkerEnv.MASTER_ADDR, None)
        if master_addr_port:
            master_addr = master_addr_port.split(":")[0]
            os.environ[HorovodEnv.RENDEZVOUS_ADDR] = master_addr
        os.environ[HorovodEnv.CONTROLLER] = "gloo"
        os.environ[HorovodEnv.CPU_OPERATIONS] = "gloo"
        domain_ip = socket.gethostbyname(socket.gethostname())
        os.environ[HorovodEnv.HOSTNAME] = domain_ip

    def notify_training_loop_status(self, status):
        self._master_client.report_training_loop_status(status)


class AllReduceController(object):
    """The controller initializes Horovod and calls the function with forward
    and backward computation using a mini-batch of data. If Horovod raise an
    exception about AllReduce, Allgather and Broadcast, the controller will
    catch the exception and re-initialize Horovod. Then, it will broadcast
    the variables and retry to call those functions.
    """

    def __init__(self, master_client, data_shard_service):
        if not hvd:
            raise RuntimeError("Horovod is not installed for AllReduce")

        self._rendezvous_manager = RendevousManager(master_client)
        self.data_shard_service = data_shard_service
        self._last_init_time = 0
        self._first_call = True
        self._need_broadcast = True

    def elastic_run(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self._init_variables_before_first_calling(func, *args, **kwargs)
            self._init_horovod_periodically()
            result = self.train_one_batch_with_retries(func, *args, **kwargs)
            self.data_shard_service.report_batch_done()
            return result

        return wrapper

    def init_horovod_locally(self):
        # Use elastic Horovod to run model locally.
        os.environ[HorovodEnv.ELASTIC] = str(1)
        hvd.init()

    def _init_variables_before_first_calling(self, func, *args, **kwargs):
        if self._first_call:
            hvd.init()
            func(*args, **kwargs)
            self._first_call = False

    def _init_horovod_periodically(self):
        """Check whether to initialize Horovod periodically in case
        that new workers join the job.
        """
        cur_time = time.time()
        if cur_time - self._last_init_time > DEFAULT_SECS_TO_CHECK_RENDEZVOUS:
            self._rendezvous_manager.init_horovod_if_needed()
            self._last_init_time = cur_time

    def _broadcast_if_needed(self):
        if self._rendezvous_manager.need_broadcast:
            logger.info("Broadcast models")
            self.broadcast()
            self._rendezvous_manager.need_broadcast = False

    def notify_train_loop_start(self):
        self._rendezvous_manager.notify_training_loop_status(
            TrainingLoopStatus.START
        )

    def notify_train_loop_end(self):
        self._rendezvous_manager.notify_training_loop_status(
            TrainingLoopStatus.END
        )

    @contextmanager
    def scope(self):
        self.notify_train_loop_start()
        yield
        self.notify_train_loop_end()

    @abstractmethod
    def train_one_batch_with_retries(self, func, *args, **kwargs):
        pass

    @abstractmethod
    def broadcast(self):
        pass
