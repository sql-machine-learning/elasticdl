import os
import unittest

import grpc
import numpy as np
import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.constants import GRPC
from elasticdl.python.common.hash_utils import string_to_id
from elasticdl.python.common.model_utils import get_model_spec
from elasticdl.python.ps.parameter_server import ParameterServer
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


_model_zoo_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../../model_zoo"
)
_test_model_zoo_path = os.path.dirname(os.path.realpath(__file__))


class PserverArgs(object):
    def __init__(
        self,
        grads_to_wait=8,
        lr_staleness_modulation=0,
        use_async=False,
        model_zoo=_test_model_zoo_path,
        model_def="test_module.custom_model",
        optimizer="optimizer",
        port=9999,
        log_level="INFO",
    ):
        self.grads_to_wait = grads_to_wait
        self.lr_staleness_modulation = lr_staleness_modulation
        self.use_async = use_async
        self.model_zoo = model_zoo
        self.model_def = model_def
        self.optimizer = optimizer
        self.port = port
        self.log_level = log_level


class WorkerMNISTTest(unittest.TestCase):
    def setUp(self):
        ports = [12345, 12346]
        self._pserver, self._channel = self._create_pserver_and_channel(ports)
        self._model_def = (
            "mnist_functional_api.mnist_functional_api.custom_model"
        )
        self._batch_size = 16

    def tearDown(self):
        for pserver in self._pserver:
            pserver.server.stop(0)

    def _create_pserver_and_channel(self, ports):
        pservers = []
        channels = []
        for port in ports:
            args = PserverArgs(grads_to_wait=1, use_async=False, port=port,)
            pserver = ParameterServer(args)
            pserver.prepare()
            pservers.append(pserver)

            addr = "localhost:%d" % port
            channel = grpc.insecure_channel(
                addr,
                options=[
                    (
                        "grpc.max_send_message_length",
                        GRPC.MAX_SEND_MESSAGE_LENGTH,
                    ),
                    (
                        "grpc.max_receive_message_length",
                        GRPC.MAX_RECEIVE_MESSAGE_LENGTH,
                    ),
                ],
            )
            channels.append(channel)
        return pservers, channels

    def test_train(self):
        images, labels = random_batch(self._batch_size)

        tf.keras.backend.clear_session()
        tf.random.set_seed(22)
        worker = Worker(
            worker_id=0,
            job_type=elasticdl_pb2.TRAINING,
            minibatch_size=self._batch_size,
            model_zoo=_model_zoo_path,
            model_def=self._model_def,
            ps_channels=self._channel,
        )
        worker._run_model_call_before_training(images)
        worker.report_variable()
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
            model_zoo=_model_zoo_path,
            model_def=self._model_def,
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

    def test_compare_train(self):
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

        acc_meter = tf.keras.metrics.Accuracy()

        tf.keras.backend.clear_session()
        tf.random.set_seed(22)

        worker = Worker(
            worker_id=0,
            job_type=elasticdl_pb2.TRAINING,
            minibatch_size=self._batch_size,
            model_zoo=_model_zoo_path,
            model_def=self._model_def,
            ps_channels=self._channel,
        )

        worker_results = []
        for step, (x, y) in enumerate(db):
            if step == 0:
                worker._run_model_call_before_training(x)
                worker.report_variable()

            worker.get_model(step, elasticdl_pb2.MINIMUM)
            w_loss, w_grads = worker.training_process_eagerly(x, y)
            worker.report_gradient(w_grads)

            if step % 20 == 0:
                worker.get_model(step, elasticdl_pb2.MINIMUM)
                for (x, y) in test_db:
                    out = worker.forward_process(x)
                    acc_meter.update_state(tf.argmax(out, axis=1), y)

                worker_results.append(
                    (float(w_loss.numpy()), float(acc_meter.result().numpy()))
                )
                acc_meter.reset_states()

            if step > 50:
                break

        tf.keras.backend.clear_session()
        tf.random.set_seed(22)
        acc_meter.reset_states()

        (
            model,
            dataset_fn,
            loss_fn,
            opt_fn,
            eval_metrics_fn,
            prediction_outputs_processor,
        ) = get_model_spec(
            model_zoo=_model_zoo_path,
            model_def=self._model_def,
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

            if step > 50:
                break

        for w, l in zip(worker_results, local_results):
            self.assertTupleEqual(w, l)


if __name__ == "__main__":
    unittest.main()
