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
import sys
import unittest

import tensorflow as tf
import torch

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.args import parse_worker_args
from elasticdl.python.tests.test_utils import (
    create_pserver,
    create_worker_args,
    get_mnist_dataset,
    get_random_batch,
)
from elasticdl.python.worker.ps_client import PSClient
from elasticdl.python.worker.worker_pytorch import WorkerPytorch
from elasticdl_client.common.constants import DistributionStrategy

sys.path.append("../")


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

    # def _create_worker(self, worker_num):
    #     args = create_worker_args(
    #         worker_num,
    #         self._batch_size,
    #         self._model_zoo_path,
    #         self._model_def,
    #         self._channels,
    #     )
    #     worker = WorkerPytorch(args, ps_client=PSClient(self._channels))
    #     self._workers.append(worker)
    
    def _create_worker(self, worker_num):
        args_list = create_worker_args(
            worker_num,
            self._batch_size,
            self._model_zoo_path,
            self._model_def,
            self._channels,
        )
        for args in args_list:
            worker = WorkerPytorch(args, ps_client=PSClient(self._channels))
            self._workers.append(worker)

    def _worker_train(self, worker_id, train_db, test_db, stop_step):
        # TODO: acc_meter with PyTorch
        acc_meter = tf.keras.metrics.Accuracy()

        worker = self._workers[worker_id]
        worker_results = []
        for step, (x, y) in enumerate(train_db):
            x = torch.from_numpy(x.numpy()).type(torch.FloatTensor)
            y = torch.from_numpy(y.numpy()).type(torch.LongTensor)
            x = torch.unsqueeze(x, dim=1)

            if step == 0:
                worker._run_model_call_before_training(x)
            worker.get_model()
            w_loss, w_grads = worker.training_process_pytorch(x, y)
            worker.report_gradient(w_grads)

            if step % 20 == 0:
                worker.get_model()
                for (x, y) in test_db:
                    x = torch.from_numpy(x.numpy()).type(torch.FloatTensor)
                    y = torch.from_numpy(y.numpy()).type(torch.LongTensor)
                    x = torch.unsqueeze(x, dim=1)

                    out = worker._model.forward(x)
                    if "mnist" in self._model_def:
                        acc_meter.update_state(
                            torch.argmax(out, dim=1).numpy(), y.numpy()
                        )
                    else:
                        break
                worker_results.append(
                    (
                        float(w_loss.detach().numpy()),
                        float(acc_meter.result().numpy()),
                    )
                )
                acc_meter.reset_states()

            if step > stop_step:
                break
        return worker_results

    def test_mnist_train(self):
        model_def = "mnist.mnist_subclass_pytorch.CustomModel"
        self._create_pserver(model_def, 2)

        # Reuse the tf style code to get dataset
        db, test_db = get_mnist_dataset(self._batch_size)

        stop_step = 20
        self._create_worker(1)
        worker_results = self._worker_train(
            0, train_db=db, test_db=test_db, stop_step=stop_step
        )

        self.assertIsInstance(worker_results[0], tuple)
        self.assertIsInstance(worker_results[0][0], float)
        self.assertIsInstance(worker_results[0][1], float)

    def test_onebatch_train_pytorch(self):
        model_def = "mnist.mnist_subclass_pytorch.CustomModel"
        self._create_pserver(model_def, 2)

        images, labels = get_random_batch(self._batch_size)
        images = torch.unsqueeze(
            torch.from_numpy(images.numpy()), dim=1
        ).float()
        labels = torch.from_numpy(labels.numpy()).type(torch.int64)
        # Focus: labels.dtype is torch.Long/torch.int64

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
        worker = WorkerPytorch(args, ps_client=PSClient(self._channels))
        worker._run_model_call_before_training(images)
        worker.get_model()
        w_loss, w_grads = worker.training_process_pytorch(images, labels)
        worker.report_gradient(w_grads)
        self.assertIsInstance(w_loss, torch.Tensor)
        self.assertIsInstance(w_grads[0], torch.Tensor)


if __name__ == "__main__":
    unittest.main()
