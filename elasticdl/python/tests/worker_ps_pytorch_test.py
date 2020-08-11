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
import random
import unittest
from threading import Thread
import numpy as np
import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.args import parse_worker_args
from elasticdl.python.common.hash_utils import int_to_id, string_to_id
from elasticdl.python.common.model_utils import get_model_spec
from elasticdl.python.ps.embedding_table import EmbeddingTable
from elasticdl.python.worker.ps_client import PSClient
from elasticdl.python.worker.worker import Worker
from elasticdl_client.common.constants import DistributionStrategy

from test_utils import (
    create_pserver,
    get_frappe_dataset,
    get_mnist_dataset,
    get_random_batch, )

import sys

sys.path.append('../')
from worker.worker_pytorch import WorkerPytorch
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(torch.utils.data.IterableDataset):
    def __init__(self, data):
        self.data_source = data

    def __iter__(self):
        return iter(self.data_source)


class WorkerPSInteractionTest(unittest.TestCase):
    def setUp(self):
        self._model_zoo_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../../../model_zoo"
        )
        self._batch_size = 16
        self._channels = []
        self._pservers = []
        self._workers = []
        self._seed_torch()

    def tearDown(self):
        for pserver in self._pservers:
            pserver.server.stop(0)

    def _seed_torch(self, seed=1029):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

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
            worker = WorkerPytorch(args, ps_client=PSClient(self._channels))
            self._workers.append(worker)

    def _worker_train(
            self, worker_id, train_db, test_db, stop_step
    ):
        worker = self._workers[worker_id]
        acc_meter = tf.keras.metrics.Accuracy()
        worker_results = []
        for step, (x, y) in enumerate(train_db):
            if step == 0:
                worker._run_model_call_before_training(x)
            worker.get_model()
            w_loss, w_grads = worker.training_process_pytorch(x, y)
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


    def _dataset_pytorch(self, dataset, batch_size):
        dataset_list = []
        for data_enum in list(dataset.as_numpy_iterator()):
            shape_0 = data_enum[0].shape[0]
            for i in range(shape_0):
                dataset_list.append((data_enum[0][i:i + 1, ...], data_enum[1][i:i + 1, ...]))

        iterable_dataset = CustomDataset(dataset_list)
        dataloader = DataLoader(dataset=iterable_dataset, batch_size=batch_size)
        return dataloader


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
        worker = Worker_pytorch(args, ps_client=PSClient(self._channels))
        worker._run_model_call_before_training(images)
        worker.get_model()
        w_loss, w_grads = worker.training_process_pytorch(images, labels)
        worker.report_gradient(w_grads)

        del worker

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
        print("finish test_compare_onebatch_train_pytorch!")

if __name__ == "__main__":
    unittest.main()
