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
import unittest
from threading import Thread

import numpy as np
import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.args import parse_worker_args
from elasticdl.python.common.hash_utils import int_to_id, string_to_id
from elasticdl.python.common.model_utils import get_model_spec
from elasticdl.python.ps.embedding_table import EmbeddingTable
from test_utils import (
    create_pserver,
    get_frappe_dataset,
    get_mnist_dataset,
    get_random_batch,)
from elasticdl.python.worker.ps_client import PSClient
from elasticdl.python.worker.worker import Worker
from elasticdl_client.common.constants import DistributionStrategy

import sys
sys.path.append('../')
from worker.worker_pytorch import Worker_pytorch
import torch

class WorkerPSInteractionTest(unittest.TestCase):
    def setUp(self):
        self._model_zoo_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../../../model_zoo"
        )
        self._batch_size = 16
        self._channels = []
        self._pservers = []
        self._workers = []

    def tearDown(self):
        for pserver in self._pservers:
            pserver.server.stop(0)

    def _create_pserver(self, model_def, num):
        self._ports, self._channels, self._pservers = create_pserver(
            self._model_zoo_path,
            model_def,
            grads_to_wait=1,
            use_async=True,
            num_ps_pods=num,
        )
        self._model_def = model_def

    def _create_worker(self, worker_num):
        for i in range(worker_num):
            tf.keras.backend.clear_session()
            tf.random.set_seed(22)
            arguments = [
                "--worker_id",
                i,
                "--job_type",
                elasticdl_pb2.TRAINING,
                "--minibatch_size",
                self._batch_size,
                "--model_zoo",
                self._model_zoo_path,
                "--model_def",
                self._model_def,
                "--distribution_strategy",
                DistributionStrategy.PARAMETER_SERVER,
            ]
            args = parse_worker_args(arguments)
            worker = Worker(args, ps_client=PSClient(self._channels))
            self._workers.append(worker)

    def _worker_train(
        self, worker_id, train_db, test_db, stop_step, use_tf_function=False
    ):
        worker = self._workers[worker_id]
        acc_meter = tf.keras.metrics.Accuracy()
        worker_results = []
        for step, (x, y) in enumerate(train_db):
            if step == 0:
                worker._run_model_call_before_training(x)

            worker.get_model()
            if use_tf_function:
                w_loss, w_grads = worker.training_process_with_acceleration(
                    x, y
                )
            else:
                w_loss, w_grads = worker.training_process_eagerly(x, y)
            worker.report_gradient(w_grads)

            if step % 20 == 0:
                worker.get_model()
                for (x, y) in test_db:
                    out = worker.forward_process(x)
                    if "mnist" in self._model_def:
                        acc_meter.update_state(tf.argmax(out, axis=1), y)
                    else:
                        out["probs"] = tf.reshape(out["probs"], [-1])
                        acc_meter.update_state(
                            tf.where(
                                out["probs"] < 0.5,
                                x=tf.zeros_like(y),
                                y=tf.ones_like(y),
                            ),
                            y,
                        )
                worker_results.append(
                    (float(w_loss.numpy()), float(acc_meter.result().numpy()))
                )
                acc_meter.reset_states()

            if step > stop_step:
                break
        return worker_results

    def test_compare_onebatch_train_pytorch(self):
        model_def = "mnist.mnist_subclass_pytorch.CustomModel"
        self._create_pserver(model_def, 2)
        images, labels = get_random_batch(self._batch_size)
        images = torch.unsqueeze(torch.from_numpy(images.numpy()), dim=1).float()
        labels = torch.from_numpy(labels.numpy()).type(torch.int64)
        
        # TODO(yunjian.lmh): test optimizer wrapper
        arguments = [
            "--worker_id",
            0,
            "--job_type",
            elasticdl_pb2.TRAINING,
            "--minibatch_size",
            self._batch_size,
            "--model_zoo",
            self._model_zoo_path,
            "--model_def",
            model_def,
            "--distribution_strategy",
            DistributionStrategy.PARAMETER_SERVER,
        ]
        args = parse_worker_args(arguments)

        tf.random.set_seed(22)

        worker = Worker_pytorch(args, ps_client=PSClient(self._channels))
        worker._run_model_call_before_training(images)
        worker.get_model()
        w_loss, w_grads = worker.training_process_pytorch(images, labels)
        worker.report_gradient(w_grads)

        tf.random.set_seed(22)

        worker = Worker_pytorch(args, ps_client=PSClient(self._channels))
        
        worker._run_model_call_before_training(images)
        worker.get_model()
        w_loss, w_grads = worker.training_process_pytorch(images, labels)
        # w_loss, w_grads = worker.training_process_eagerly(images, labels)
        worker.report_gradient(w_grads)

        # tf.keras.backend.clear_session()
        tf.random.set_seed(22)

        (
            model,
            dataset_fn,
            loss_fn,
            opt_fn,
            eval_metrics_fn,
            prediction_outputs_processor,
            create_data_reader_fn,
            callback_list,
        ) = get_model_spec(
            model_zoo=self._model_zoo_path,
            model_def=model_def,
            dataset_fn="dataset_fn",
            model_params=None,
            loss="loss",
            optimizer="optimizer",
            eval_metrics_fn="eval_metrics_fn",
            prediction_outputs_processor="PredictionOutputsProcessor",
            custom_data_reader="custom_data_reader",
            callbacks="callbacks",
        )

        # opt_fn(model).zero_grad()
        for name, parms in model.named_parameters():
            parms = torch.zeros_like(parms)

        output = model.forward(images)
        labels = torch.reshape(labels, [-1])
        loss = loss_fn(labels, output)
        loss.backward()
        for name, parms in model.named_parameters():
            if parms.requires_grad:
                ps_id = string_to_id(name, len(self._channels))
                ps_v = self._pservers[ps_id].parameters.get_non_embedding_param(
                    name)
                np.testing.assert_array_equal(ps_v.numpy(), parms.data.numpy())


if __name__ == "__main__":
    unittest.main()
