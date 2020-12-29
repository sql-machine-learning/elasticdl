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

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.framework.errors_impl import UnknownError

from elasticai_api.common.base_controller import (
    DEFAULT_MAX_ALLREDUCE_RETRY_NUM,
    DEFAULT_SECS_TO_CHECK_RENDEZVOUS,
    RendevousManager,
)
from elasticai_api.common.constants import TrainingLoopStatus
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.worker.trainer import Trainer

try:
    from horovod.tensorflow.functions import broadcast_variables
    import horovod.tensorflow as hvd

except ImportError:
    hvd = None


class AllReduceTrainer(Trainer):
    def __init__(self, master_client, model):
        if not hvd:
            raise RuntimeError("Horovod is not installed for AllReduce")
        self._rendezvous_manager = RendevousManager(master_client)
        self._model = model
        self._loss = model.loss
        self._need_broadcast = True
        self._var_created = False
        self._optimizer = model.optimizer
        self._last_init_time = 0

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
        cur_time = time.time()
        if cur_time - self._last_init_time > DEFAULT_SECS_TO_CHECK_RENDEZVOUS:
            self._rendezvous_manager.init_horovod_if_needed()
            self._last_init_time = cur_time

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

    def notify_training_loop_start(self):
        self._rendezvous_manager.notify_training_loop_status(
            TrainingLoopStatus.START
        )

    def notify_training_loop_end(self):
        self._rendezvous_manager.notify_training_loop_status(
            TrainingLoopStatus.END
        )
