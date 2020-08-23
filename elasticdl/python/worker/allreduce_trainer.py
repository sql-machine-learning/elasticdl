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

import tensorflow as tf
from tensorflow.python.framework.errors_impl import UnknownError

from elasticdl.python.common.constants import HorovodEnv
from elasticdl.python.common.log_utils import default_logger as logger

try:
    from horovod.tensorflow.functions import broadcast_variables
    import horovod.tensorflow as hvd
except ImportError:
    hvd = None


# The default maximum number of retries for allreduce operation
# if allreduce-based distributed training strategy is used.
DEFAULT_MAX_ALLREDUCE_RETRY_NUM = 5


class AllReduceTrainer(object):
    def __init__(self, master_client, master_addr, model, loss_fn, optimizer):
        if not hvd:
            raise RuntimeError("Horovod is not installed for AllReduce")
        self._master_client = master_client
        self._rendezvous_addr = master_addr
        self._model = model
        self._loss = loss_fn
        self._optimizer = optimizer
        self._rendezvous_id = None
        self._need_broadcast = True
        self._var_created = False
        self.set_horovod_env()

    @tf.function
    def _training_process(self, features, labels):
        with tf.GradientTape() as tape:
            outputs = self._model.call(features, training=True)
            loss = self._loss(labels, outputs)
            if self._model.losses:
                loss += tf.math.add_n(self._model.losses)

        tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(loss, self._model.trainable_variables)
        # Take care of the order of grads and vars if worker modifies
        # `_non_embed_vars` during training.
        self._optimizer.apply_gradients(
            zip(grads, self._model.trainable_variables)
        )
        return loss

    def training_process_with_fault_tolerance(self, features, labels):
        if not self._var_created:
            self._run_model_call_locally(features, labels)

        for _ in range(DEFAULT_MAX_ALLREDUCE_RETRY_NUM + 1):
            try:
                if self._need_broadcast:
                    self._broadcast_model()
                    self._need_broadcast = False
                loss = self._training_process(features, labels)
                version = self._optimizer.iterations.numpy()
                return version, loss
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
                    self.init_horovod_if_needed()

    def init_horovod_if_needed(self):
        for _ in range(DEFAULT_MAX_ALLREDUCE_RETRY_NUM):
            rank_response = self._master_client.get_comm_rank()
            if rank_response.rank_id < 0:
                logger.warning(
                    "The master has not added the worker host into "
                    "rendezvous yet. Retrying to get rank"
                )
                time.sleep(5)
            else:
                break

        # If the rendezvous from master is unequal to self._rendezvous_id,
        # the worker should rebuild the communication because the master
        # has updated the communication group.
        if rank_response.rendezvous_id != self._rendezvous_id:
            os.environ[HorovodEnv.RENDEZVOUS_PORT] = str(
                rank_response.rendezvous_port
            )
            os.environ[HorovodEnv.RANK] = str(rank_response.rank_id)
            os.environ[HorovodEnv.SIZE] = str(rank_response.world_size)
            hvd.shutdown()
            hvd.init()
            self._rendezvous_id = rank_response.rendezvous_id
            self._need_broadcast = True

    def set_horovod_env(self):
        if self._rendezvous_addr:
            os.environ[HorovodEnv.RENDEZVOUS_ADDR] = self._rendezvous_addr
        os.environ[HorovodEnv.CONTROLLER] = "gloo"
        os.environ[HorovodEnv.CPU_OPERATIONS] = "gloo"
        os.environ[HorovodEnv.HOSTNAME] = "master"

    def _broadcast_model(self):
        broadcast_variables(self._model.variables, root_rank=0)
        broadcast_variables(self._optimizer.variables(), root_rank=0)

    def _run_model_call_locally(self, features, labels):
        """Call `self._model.call` locally to Create variables of the model
        and optimizer. Because we should have variables before broadcasting.
        """
        with tf.GradientTape() as tape:
            outputs = self._model.call(features)
            loss = self._loss(labels, outputs)

        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(
            zip(grads, self._model.trainable_variables)
        )
        self._var_created = True
