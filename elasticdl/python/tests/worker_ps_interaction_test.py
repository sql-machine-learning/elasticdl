import os
import unittest
from pathlib import Path

import grpc
import numpy as np
import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.args import parse_worker_args
from elasticdl.python.common.constants import GRPC
from elasticdl.python.common.grpc_utils import build_channel
from elasticdl.python.common.hash_utils import int_to_id, string_to_id
from elasticdl.python.common.model_utils import get_model_spec
from elasticdl.python.data.recordio_gen.frappe_recordio_gen import (
    load_raw_data,
)
from elasticdl.python.ps.embedding_table import EmbeddingTable
from elasticdl.python.ps.parameter_server import ParameterServer
from elasticdl.python.tests.test_utils import PserverArgs
from elasticdl.python.worker.worker import Worker


def random_batch(batch_size):
    shape = (28, 28)
    shape = (batch_size,) + shape
    num_classes = 10
    images = tf.random.uniform(shape)
    labels = tf.random.uniform(
        [batch_size], minval=0, maxval=num_classes, dtype=tf.int32
    )
    return images, labels


class WorkerPSInteractionTest(unittest.TestCase):
    def setUp(self):
        self._model_zoo_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../../../model_zoo"
        )
        self._model_def = (
            "mnist_functional_api.mnist_functional_api.custom_model"
        )
        self._batch_size = 16
        self._ports = [12345, 12346]
        self._pserver, self._channel = self._create_pserver_and_channel(
            self._ports
        )

    def tearDown(self):
        for pserver in self._pserver:
            pserver.server.stop(0)

    def _create_pserver_and_channel(self, ports):
        pservers = []
        channels = []
        for port in ports:
            args = PserverArgs(
                grads_to_wait=1,
                use_async=False,
                port=port,
                model_zoo=self._model_zoo_path,
                model_def=self._model_def,
            )
            pserver = ParameterServer(args)
            pserver.prepare()
            pservers.append(pserver)

            addr = "localhost:%d" % port
            channel = build_channel(addr)
            channels.append(channel)
        return pservers, channels

    def _restart_pserver(self):
        # Stop first
        self.tearDown()
        # Start again
        self._pserver, self._channel = self._create_pserver_and_channel(
            self._ports
        )

    def test_worker_pull_embedding(self):
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
            self._model_def,
            "--distribution_strategy",
            "ParameterServerStrategy",
        ]
        args = parse_worker_args(arguments)
        worker = Worker(args, ps_channels=self._channel)

        # Test lookup embedding vectors that do not exist
        layers = ["test-2", "test-2-slot"]
        ids = [3, 5, 1, 6, 10, 2, 1, 2, 4, 7, 9]
        embedding_table_args = [
            (layers[0], 8, "uniform", False),
            (layers[1], 8, 3.3, True),
        ]

        # initialize embedding table object
        for pserver in self._pserver:
            for layer, table_args in zip(layers, embedding_table_args):
                pserver.parameters.embedding_params[layer] = EmbeddingTable(
                    *table_args
                )

        result_dict = {}
        for layer in layers:
            embedding = worker.pull_embedding_vector(layer, ids)
            result_dict[layer] = embedding

        for layer in layers:
            expected_result = []
            for embedding_id in ids:
                ps_id = int_to_id(embedding_id, len(self._pserver))
                table = self._pserver[ps_id].parameters.embedding_params[layer]
                expected_result.append(table.get([embedding_id]))
            expected_result = np.concatenate(expected_result)
            self.assertTrue(np.allclose(expected_result, result_dict[layer]))

    def test_compare_onebatch_train(self):
        images, labels = random_batch(self._batch_size)
        # TODO(yunjian.lmh): test optimizer wrapper
        tf.keras.backend.clear_session()
        tf.random.set_seed(22)
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
            "mnist_functional_api.mnist_functional_api.custom_model",
            "--distribution_strategy",
            "ParameterServerStrategy",
        ]
        args = parse_worker_args(arguments)

        worker = Worker(args, ps_channels=self._channel)
        worker._run_model_call_before_training(images)
        worker.get_model(0, elasticdl_pb2.MINIMUM)
        w_loss, w_grads = worker.training_process_eagerly(images, labels)
        worker.report_gradient(w_grads)

        tf.keras.backend.clear_session()
        tf.random.set_seed(22)
        (
            model,
            dataset_fn,
            loss_fn,
            opt_fn,
            eval_metrics_fn,
            prediction_outputs_processor,
        ) = get_model_spec(
            model_zoo=self._model_zoo_path,
            model_def=(
                "mnist_functional_api.mnist_functional_api.custom_model"
            ),
            dataset_fn="dataset_fn",
            model_params=None,
            loss="loss",
            optimizer="optimizer",
            eval_metrics_fn="eval_metrics_fn",
            prediction_outputs_processor="PredictionOutputsProcessor",
        )

        with tf.GradientTape() as tape:
            output = model.call(images, training=True)
            labels = tf.reshape(labels, [-1])
            loss = loss_fn(output, labels)
        grads = tape.gradient(loss, model.trainable_variables)
        opt_fn().apply_gradients(zip(grads, model.trainable_variables))

        for v in model.trainable_variables:
            ps_id = string_to_id(v.name, len(self._channel))
            ps_v = self._pserver[ps_id].parameters.get_non_embedding_param(
                v.name
            )
            np.testing.assert_array_equal(ps_v.numpy(), v.numpy())

    def _worker_train(self, train_db, test_db, dataset, stop_step):
        if dataset == "mnist":
            model_def = (
                "mnist_functional_api.mnist_functional_api.custom_model"
            )
        elif dataset == "frappe":
            model_def = (
                "deepfm_functional_api.deepfm_functional_api.custom_model"
            )
        else:
            raise ValueError("dataset %s is not supported", dataset)
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
            "ParameterServerStrategy",
        ]
        args = parse_worker_args(arguments)

        worker = Worker(args, ps_channels=self._channel)
        acc_meter = tf.keras.metrics.Accuracy()
        worker_results = []
        for step, (x, y) in enumerate(train_db):
            if step == 0:
                worker._run_model_call_before_training(x)

            worker.get_model(step, elasticdl_pb2.MINIMUM)

            w_loss, w_grads = worker.training_process_eagerly(x, y)
            worker.report_gradient(w_grads)

            if step % 20 == 0:
                worker.get_model(step, elasticdl_pb2.MINIMUM)
                for (x, y) in test_db:
                    out = worker.forward_process(x)
                    if dataset == "mnist":
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

    def test_compare_mnist_train(self):
        (
            (x_train, y_train),
            (x_test, y_test),
        ) = tf.keras.datasets.mnist.load_data()
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255.0
        y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)

        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255.0
        y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

        db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        db = db.batch(self._batch_size).repeat(10)
        test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_db = test_db.batch(self._batch_size)

        tf.keras.backend.clear_session()
        tf.random.set_seed(22)
        stop_step = 20
        worker_results = self._worker_train(
            train_db=db, test_db=test_db, dataset="mnist", stop_step=stop_step
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
        ) = get_model_spec(
            model_zoo=self._model_zoo_path,
            model_def=(
                "mnist_functional_api.mnist_functional_api.custom_model"
            ),
            dataset_fn="dataset_fn",
            model_params=None,
            loss="loss",
            optimizer="optimizer",
            eval_metrics_fn="eval_metrics_fn",
            prediction_outputs_processor="PredictionOutputsProcessor",
        )
        local_results = []
        for step, (x, y) in enumerate(db):

            with tf.GradientTape() as tape:
                out = model.call(x, training=True)
                ll = loss_fn(out, y)
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

    def test_deepfm_train(self):
        tf.keras.backend.clear_session()
        tf.random.set_seed(22)
        home = str(Path.home())

        class TmpArgs(object):
            def __init__(self, data):
                self.data = data

        args = TmpArgs(data=home + "/.keras/datasets/")

        x_train, y_train, x_val, y_val, x_test, y_test = load_raw_data(args)
        x_train = tf.convert_to_tensor(x_train, dtype=tf.int64)
        x_test = tf.convert_to_tensor(x_test, dtype=tf.int64)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.int64)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.int64)

        db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        db = db.batch(self._batch_size).repeat(10)
        test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_db = test_db.batch(self._batch_size)

        worker_results = self._worker_train(
            train_db=db, test_db=test_db, dataset="frappe", stop_step=100
        )
        acc = max([r[1] for r in worker_results])
        self.assertLess(0.6, acc)

    def test_restart_ps(self):
        num_data = 8
        training_data = [
            random_batch(self._batch_size) for _ in range(num_data)
        ]
        workers = []
        for w in range(2):
            self._restart_pserver()
            tf.keras.backend.clear_session()
            tf.random.set_seed(22)
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
                self._model_def,
                "--distribution_strategy",
                "ParameterServerStrategy",
            ]
            args = parse_worker_args(arguments)

            worker = Worker(args, ps_channels=self._channel)
            workers.append(worker)
            worker._run_model_call_before_training(training_data[0][0])
            for i in range(num_data):
                worker.get_model(0, elasticdl_pb2.MINIMUM)
                w_loss, w_grads = worker.training_process_eagerly(
                    training_data[i][0], training_data[i][1]
                )
                worker.report_gradient(w_grads)
                if w == 1 and i == 3:
                    # Restart ps for the 2nd worker at i==3
                    self._restart_pserver()
                    # `report_variable` will be called in `get_model` to
                    # initialize variables on ps with worker variables
                    worker.get_model(0, elasticdl_pb2.MINIMUM)
                    # send the grads again as these grads are not applied
                    # on worker variables
                    worker.report_gradient(w_grads)

        for var_name in workers[0]._non_embed_vars:
            np.testing.assert_array_equal(
                workers[0]._non_embed_vars[var_name].numpy(),
                workers[1]._non_embed_vars[var_name].numpy(),
            )


if __name__ == "__main__":
    unittest.main()
