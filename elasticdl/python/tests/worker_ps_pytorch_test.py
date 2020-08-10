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
<<<<<<< HEAD
=======
# from elasticdl.python.worker.my_worker_pytorch import Worker_pytorch
>>>>>>> 9c0c4f8b99139d88f8688a1a2a27e1f9a8351dfb
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
<<<<<<< HEAD
        model_def = "mnist.mnist_subclass_pytorch.CustomModel"
        self._create_pserver(model_def, 2)
        images, labels = get_random_batch(self._batch_size)
        images = torch.unsqueeze(torch.from_numpy(images.numpy()), dim=1).float()
        labels = torch.from_numpy(labels.numpy()).type(torch.int64)
=======
        print("===========test_compare_onebatch_train begin===========")
        # model_def = "mnist.mnist_functional_api.custom_model"
        model_def = "mnist.mnist_subclass_pytorch.CustomModel"
        self._create_pserver(model_def, 2)
        images, labels = get_random_batch(self._batch_size)
        images = torch.from_numpy(images.numpy())
        # labels = torch.from_numpy(labels.numpy()).type(torch.long)
        # images = torch.unsqueeze(images, dim=1).type(torch.long)
        labels = torch.from_numpy(labels.numpy()).type(torch.int64)
        images = torch.unsqueeze(images, dim=1).float()
        print("images:",images.size(),images.dtype)  # torch.Size([16, 1, 28, 28])
        print("labels:",labels.size(),labels.dtype)  # torch.Size([16])
>>>>>>> 9c0c4f8b99139d88f8688a1a2a27e1f9a8351dfb
        
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

<<<<<<< HEAD
        tf.random.set_seed(22)

        worker = Worker_pytorch(args, ps_client=PSClient(self._channels))
        worker._run_model_call_before_training(images)
        worker.get_model()
        w_loss, w_grads = worker.training_process_pytorch(images, labels)
        worker.report_gradient(w_grads)

        tf.random.set_seed(22)
=======
        # tf.keras.backend.clear_session()
        tf.random.set_seed(22)

        # worker = Worker(args, ps_client=PSClient(self._channels))
        worker = Worker_pytorch(args, ps_client=PSClient(self._channels))
        
        worker._run_model_call_before_training(images)
        worker.get_model()
        w_loss, w_grads = worker.training_process_pytorch(images, labels)
        # w_loss, w_grads = worker.training_process_eagerly(images, labels)
        worker.report_gradient(w_grads)

        # tf.keras.backend.clear_session()
        tf.random.set_seed(22)

>>>>>>> 9c0c4f8b99139d88f8688a1a2a27e1f9a8351dfb
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
<<<<<<< HEAD
        # opt_fn(model).zero_grad()
        for name, parms in model.named_parameters():
            parms = torch.zeros_like(parms)
=======
        """
        with tf.GradientTape() as tape:
            output = model.call(images, training=True)
            labels = tf.reshape(labels, [-1])
            loss = loss_fn(labels, output)
        grads = tape.gradient(loss, model.trainable_variables)
        opt_fn().apply_gradients(zip(grads, model.trainable_variables))
        """
        opt_fn(model).zero_grad()

>>>>>>> 9c0c4f8b99139d88f8688a1a2a27e1f9a8351dfb
        output = model.forward(images)
        labels = torch.reshape(labels, [-1])
        loss = loss_fn(labels, output)
        loss.backward()

<<<<<<< HEAD
        for name, parms in model.named_parameters():
            if parms.requires_grad:
                ps_id = string_to_id(name, len(self._channels))
                ps_v = self._pservers[ps_id].parameters.get_non_embedding_param(
                    name)
                np.testing.assert_array_equal(ps_v.numpy(), parms.data.numpy())
=======
        loss_tmp = []
        for name, parms in model.named_parameters():
            if parms.requires_grad:
                ps_id = string_to_id(name, len(self._channels))
                # loss_tmp[name] = parms
                ps_v = self._pservers[ps_id].parameters.get_non_embedding_param(
                    name)
                np.testing.assert_array_equal(ps_v.numpy(), parms.data.numpy())
        print("===========test_compare_onebatch_train end===========")  
        


    """
    def test_compare_mnist_train(self):
        print("===========test_compare_mnist_train begin===========")
        model_def = "mnist.mnist_functional_api.custom_model"
        self._create_pserver(model_def, 2)
        db, test_db = get_mnist_dataset(self._batch_size)
        stop_step = 20

        self._create_worker(1)
        worker_results = self._worker_train(
            0, train_db=db, test_db=test_db, stop_step=stop_step
        )

        tf.keras.backend.clear_session()
        tf.random.set_seed(22)

        acc_meter = tf.keras.metrics.Accuracy()

        (
            model,
            dataset_fn,
            loss_fn,
            opt_fn,
            eval_metrics_fn,
            prediction_outputs_processor,
            create_data_reader_fn,
            callbacks_list,
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
        local_results = []
        for step, (x, y) in enumerate(db):
            with tf.GradientTape() as tape:
                out = model.call(x, training=True)
                ll = loss_fn(y, out)
            grads = tape.gradient(ll, model.trainable_variables)
            opt_fn().apply_gradients(zip(grads, model.trainable_variables))

            if step % 20 == 0:
                for (x, y) in test_db:
                    out = model.call(x, training=False)
                    acc_meter.update_state(tf.argmax(out, axis=1), y)

                local_results.append(
                    (float(ll.numpy()), float(acc_meter.result().numpy()))
                )
                acc_meter.reset_states()

            if step > stop_step:
                break

        for w, l in zip(worker_results, local_results):
            self.assertTupleEqual(w, l)
        print("===========test_compare_mnist_train end===========")
    """

>>>>>>> 9c0c4f8b99139d88f8688a1a2a27e1f9a8351dfb

if __name__ == "__main__":
    unittest.main()
