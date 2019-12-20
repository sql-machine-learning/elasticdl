import os
import unittest
from threading import Thread

import numpy as np
import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.args import parse_worker_args
from elasticdl.python.common.constants import DistributionStrategy
from elasticdl.python.common.hash_utils import int_to_id, string_to_id
from elasticdl.python.common.model_utils import get_model_spec
from elasticdl.python.ps.embedding_table import EmbeddingTable
from elasticdl.python.tests.test_utils import (
    create_pserver,
    get_frappe_dataset,
    get_mnist_dataset,
    get_random_batch,
)
from elasticdl.python.worker.worker import Worker


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

    def _reset_pserver(self):
        for ps in self._pservers:
            ps.parameters.reset()

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
            worker = Worker(args, ps_channels=self._channels)
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

    def _test_deepfm_train(self, num_ps, num_worker, stop_step):
        model_def = "deepfm_functional_api.deepfm_functional_api.custom_model"
        self._create_pserver(model_def, num_ps)
        db, test_db = get_frappe_dataset(self._batch_size)

        self._create_worker(num_worker)
        threads = []
        for w in range(num_worker):
            t = Thread(
                target=self._worker_train, args=(w, db, test_db, stop_step)
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def test_worker_pull_embedding(self):
        model_def = "mnist_functional_api.mnist_functional_api.custom_model"
        self._create_pserver(model_def, 2)
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
        worker = Worker(args, ps_channels=self._channels)

        # Test lookup embedding vectors that do not exist
        layers = ["test-2", "test-2-slot"]
        ids = [3, 5, 1, 6, 10, 2, 1, 2, 4, 7, 9]
        embedding_table_args = [
            (layers[0], 8, "uniform", False),
            (layers[1], 8, 3.3, True),
        ]

        # initialize embedding table object
        for pserver in self._pservers:
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
                ps_id = int_to_id(embedding_id, len(self._pservers))
                table = self._pservers[ps_id].parameters.embedding_params[
                    layer
                ]
                expected_result.append(table.get([embedding_id]))
            expected_result = np.concatenate(expected_result)
            self.assertTrue(np.allclose(expected_result, result_dict[layer]))

    def test_compare_onebatch_train(self):
        model_def = "mnist_functional_api.mnist_functional_api.custom_model"
        self._create_pserver(model_def, 2)
        images, labels = get_random_batch(self._batch_size)
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

        tf.keras.backend.clear_session()
        tf.random.set_seed(22)

        worker = Worker(args, ps_channels=self._channels)
        worker._run_model_call_before_training(images)
        worker.get_model()
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
            model_def=model_def,
            dataset_fn="dataset_fn",
            model_params=None,
            loss="loss",
            optimizer="optimizer",
            eval_metrics_fn="eval_metrics_fn",
            prediction_outputs_processor="PredictionOutputsProcessor",
            custom_data_reader="custom_data_reader",
        )

        with tf.GradientTape() as tape:
            output = model.call(images, training=True)
            labels = tf.reshape(labels, [-1])
            loss = loss_fn(labels, output)
        grads = tape.gradient(loss, model.trainable_variables)
        opt_fn().apply_gradients(zip(grads, model.trainable_variables))

        for v in model.trainable_variables:
            ps_id = string_to_id(v.name, len(self._channels))
            ps_v = self._pservers[ps_id].parameters.get_non_embedding_param(
                v.name
            )
            np.testing.assert_array_equal(ps_v.numpy(), v.numpy())

    def test_compare_mnist_train(self):
        model_def = "mnist_functional_api.mnist_functional_api.custom_model"
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

    def test_deepfm_train(self):
        model_def = "deepfm_functional_api.deepfm_functional_api.custom_model"
        self._create_pserver(model_def, 2)
        db, test_db = get_frappe_dataset(self._batch_size)
        self._create_worker(1)
        worker_results = self._worker_train(
            0, train_db=db, test_db=test_db, stop_step=100
        )
        acc = max([r[1] for r in worker_results])
        self.assertLess(0.65, acc)

    def test_deepfm_two_worker_train(self):
        num_ps = 2
        num_worker = 2
        stop_step = 10
        self._test_deepfm_train(num_ps, num_worker, stop_step)

    def test_deepfm_four_worker_train(self):
        num_ps = 4
        num_worker = 1
        stop_step = 10
        self._test_deepfm_train(num_ps, num_worker, stop_step)

    def test_restart_ps(self):
        model_def = "mnist_functional_api.mnist_functional_api.custom_model"
        num_data = 8
        training_data = [
            get_random_batch(self._batch_size) for _ in range(num_data)
        ]
        workers = []
        self._create_pserver(model_def, 2)
        for w in range(2):
            self._reset_pserver()
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
            tf.keras.backend.clear_session()
            tf.random.set_seed(22)
            worker = Worker(args, ps_channels=self._channels)
            workers.append(worker)
            worker._run_model_call_before_training(training_data[0][0])
            for i in range(num_data):
                worker.get_model()
                w_loss, w_grads = worker.training_process_eagerly(
                    training_data[i][0], training_data[i][1]
                )
                worker.report_gradient(w_grads)
                if w == 1 and i == 3:
                    # Restart ps for the 2nd worker at i==3
                    # self._restart_pserver(model_def)
                    self._reset_pserver()
                    # `report_variable` will be called in `get_model` to
                    # initialize variables on ps with worker variables
                    worker.get_model()
                    # send the grads again as these grads are not applied
                    # on worker variables
                    worker.report_gradient(w_grads)

        for var_name in workers[0]._non_embed_vars:
            np.testing.assert_array_equal(
                workers[0]._non_embed_vars[var_name].numpy(),
                workers[1]._non_embed_vars[var_name].numpy(),
            )

    def test_train_acceleration_with_embedding(self):
        model_def = "deepfm_functional_api.deepfm_functional_api.custom_model"
        self._create_pserver(model_def, 2)
        db, test_db = get_frappe_dataset(self._batch_size)
        self._create_worker(1)
        worker_results = self._worker_train(
            0,
            train_db=db,
            test_db=test_db,
            stop_step=100,
            use_tf_function=True,
        )
        acc = max([r[1] for r in worker_results])
        self.assertLess(0.65, acc)


if __name__ == "__main__":
    unittest.main()
