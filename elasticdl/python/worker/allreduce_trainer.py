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
from functools import wraps

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.framework.errors_impl import UnknownError

from elasticdl.python.common.constants import HorovodEnv
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.worker.trainer import Trainer

try:
    from horovod.tensorflow.functions import broadcast_variables
    import horovod.tensorflow as hvd
except ImportError:
    hvd = None


# The default maximum number of retries for allreduce operation
# if allreduce-based distributed training strategy is used.
DEFAULT_MAX_ALLREDUCE_RETRY_NUM = 5
DEFAULT_STEPS_TO_CHECK_RENDEZVOUS = 20


class RendevousManager(object):
    def __init__(self, master_client, master_addr):
        self.need_broadcast = True
        self._master_client = master_client
        self._rendezvous_addr = master_addr
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
            # Not using Horovod elastic feature in init, but need it for
            # allreduce to call allreduce op when size=1.
            os.environ[HorovodEnv.ELASTIC] = str(0)
            hvd.shutdown()
            hvd.init()
            os.environ[HorovodEnv.ELASTIC] = str(1)
            self._rendezvous_id = rank_response.rendezvous_id
            self.need_broadcast = True

    def _set_horovod_env(self):
        if self._rendezvous_addr:
            os.environ[HorovodEnv.RENDEZVOUS_ADDR] = self._rendezvous_addr
        os.environ[HorovodEnv.CONTROLLER] = "gloo"
        os.environ[HorovodEnv.CPU_OPERATIONS] = "gloo"
        domain_ip = socket.gethostbyname(socket.gethostname())
        os.environ[HorovodEnv.HOSTNAME] = domain_ip


class AllReduceTrainer(Trainer):
    def __init__(self, master_client, master_addr, model):
        if not hvd:
            raise RuntimeError("Horovod is not installed for AllReduce")
        self._rendezvous_manager = RendevousManager(master_client, master_addr)
        self._model = model
        self._loss = model.loss
        self._need_broadcast = True
        self._var_created = False
        self._optimizer = model.optimizer

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

    def train_minibatch(self, features, labels, train_with_local_model=False):
        self._check_new_communication_world()

        for _ in range(DEFAULT_MAX_ALLREDUCE_RETRY_NUM + 1):
            try:
                if self._rendezvous_manager.need_broadcast:
                    self._broadcast_model()
                    self._rendezvous_manager.need_broadcast = False
                loss = self._training_process(features, labels)
                version = self._optimizer.iterations.numpy()
                return True, version, loss
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

    def _check_new_communication_world(self):
        """"Check periodically whether new workers join the job
        and re-initialize Horovod if True.
        """
        iter_steps = self._optimizer.iterations.numpy()

        if iter_steps % DEFAULT_STEPS_TO_CHECK_RENDEZVOUS == 0:
            self._rendezvous_manager.init_horovod_if_needed()

    def _broadcast_model(self):
        broadcast_variables(self._model.variables, root_rank=0)
        broadcast_variables(self._optimizer.variables(), root_rank=0)

    def init_variables_if_need(self, features, labels):
        if not self._var_created:
            self._run_model_call_locally(features, labels)
        self._var_created = True

    def _run_model_call_locally(self, features, labels):
        """Call `self._model.call` locally to create variables of the model
        and optimizer. Because we should have variables before broadcasting.
        """
        with tf.GradientTape() as tape:
            outputs = self._model.call(features)
            loss = self._loss(labels, outputs)

        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(
            zip(grads, self._model.trainable_variables)
        )
        # TODO: Handle the case that the model is initialized from a checkpoint
        K.set_value(self._optimizer.iterations, 0)
        self._var_created = True

    def export_saved_model(self, model_path):
        if not model_path:
            return
        if hvd.rank() == 0:
            self._model.save(
                model_path, overwrite=True, include_optimizer=False
            )

    def get_model_version(self):
        return self._optimizer.iterations.numpy()


class ElasticAllReduceController(object):
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

            for _ in range(DEFAULT_MAX_ALLREDUCE_RETRY_NUM + 1):
                try:
                    self._broadcast_if_needed()
                    self._step += 1
                    return func(*args, **kwargs)
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

        return wrapper

    def _init_variables_before_first_calling(self, func, *args, **kwargs):
        if self._first_call:
            hvd.init()
            func(*args, **kwargs)
            self._first_call = False

    def _init_horovod_periodically(self):
        if self._step % 20 == 0:
            self._rendezvous_manager.init_horovod_if_needed()

    def _broadcast_if_needed(self):
        if self._rendezvous_manager.need_broadcast:
            logger.info("Broadcast models")
            self.broadcast()
            self._rendezvous_manager.need_broadcast = False

    @abstractmethod
    def broadcast(self):
        pass


class ElasticTensorFlowV2Controller(ElasticAllReduceController):
    def __init__(self, master_client, master_addr):
        super(ElasticTensorFlowV2Controller, self).__init__(
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
