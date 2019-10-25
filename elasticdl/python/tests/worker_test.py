import copy
import os
import unittest

import mock
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Flatten

from elasticdl.python.common.constants import JobType
from elasticdl.python.elasticdl.layers.embedding import Embedding
from elasticdl.python.master.embedding_service import EmbeddingService
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.tests import test_call_back
from elasticdl.python.tests.in_process_master import InProcessMaster
from elasticdl.python.tests.mock_kv_store import MockKvStore
from elasticdl.python.tests.test_call_back import BaseCallback
from elasticdl.python.tests.test_utils import (
    DatasetName,
    distributed_train_and_evaluate,
)
from elasticdl.python.worker.worker import Worker

_model_zoo_path = os.path.dirname(os.path.realpath(__file__))


class CustomModel(tf.keras.Model):
    def __init__(self, output_dim=16):
        super(CustomModel, self).__init__(name="CustomModel")
        self.output_dim = output_dim
        self.embedding_1 = Embedding(output_dim)
        self.embedding_2 = Embedding(output_dim)
        self.concat = Concatenate()
        self.dense = Dense(1)
        self.flatten = Flatten()

    def call(self, inputs, training=False):
        f1 = self.embedding_1(inputs["f1"])
        f2 = self.embedding_2(inputs["f2"])
        x = self.concat([f1, f2])
        x = self.dense(x)
        return self.flatten(x)


class CheckRetryCallback(BaseCallback):
    """Checks the retry functionality of workers.

    When workers report Gradient or evaluation metrics, this callback
    adds 1 to master's model version. The master rejects the report request and
    workers retry.
    """

    def __init__(self, master, worker):
        super(CheckRetryCallback, self).__init__(
            master,
            worker,
            call_times=[
                test_call_back.ON_REPORT_GRADIENT_BEGIN,
                test_call_back.ON_REPORT_EVALUATION_METRICS_BEGIN,
            ],
        )

    def __call__(self):
        if 2 < self._master._version < 80:
            self._master._version += 1


class CheckWorkerModelCallback(BaseCallback):
    """Checks worker model parameters.

    Before master updating model parameters, master and workers should have
    same model parameters if `master._grad_n=0`.
    """

    def __init__(self, master, worker):
        super(CheckWorkerModelCallback, self).__init__(
            master,
            worker,
            call_times=[test_call_back.ON_REPORT_GRADIENT_BEGIN],
        )

    def __call__(self):
        if self._master._grad_n:
            return
        worker_var_n = len(self._worker._non_embed_vars)
        master_var_n = len(self._master._model)
        if worker_var_n != master_var_n:
            raise RuntimeError(
                "The number of non-embedding variables in worker %d differs "
                "from the number of `_model` variables in master %d."
                % (worker_var_n, master_var_n)
            )
        for var in self._worker._non_embed_vars.values():
            if not np.isclose(
                var.numpy(), self._master._model[var.name].numpy()
            ).all():
                raise RuntimeError(
                    "The value of variable %s in worker differs from its "
                    "value in master." % (var.name)
                )


class WorkerTest(unittest.TestCase):
    def test_distributed_train_tf_example(self):
        distributed_train_and_evaluate(
            1,
            _model_zoo_path,
            "test_module.custom_model",
            training=True,
            dataset_name=DatasetName.TEST_MODULE,
            callback_classes=[CheckRetryCallback],
        )

    def test_distributed_evaluate_tf_example(self):
        distributed_train_and_evaluate(
            1,
            _model_zoo_path,
            "test_module.custom_model",
            training=False,
            dataset_name=DatasetName.TEST_MODULE,
            callback_classes=[CheckRetryCallback],
        )

    def test_distributed_train_get_model_steps(self):
        distributed_train_and_evaluate(
            1,
            _model_zoo_path,
            "test_module.custom_model",
            training=True,
            dataset_name=DatasetName.TEST_MODULE,
            callback_classes=[CheckWorkerModelCallback],
            use_async=True,
            get_model_steps=4,
        )

    def test_embedding_layer(self):
        worker = Worker(
            1,
            JobType.TRAINING_ONLY,
            32,
            _model_zoo_path,
            model_def="embedding_test_module.EdlEmbeddingModel",
            channel=None,
        )
        self.assertTrue(len(worker._embedding_layers) == 2)

    def test_train_acceleration_with_embedding(self):
        kv_store = MockKvStore()
        model_inst = CustomModel()
        master = MasterServicer(
            2,
            2,
            tf.optimizers.SGD(0.1),
            None,
            init_var=model_inst.trainable_variables,
            checkpoint_filename_for_init=None,
            checkpoint_service=None,
            evaluation_service=None,
        )
        worker = Worker(
            1,
            JobType.TRAINING_ONLY,
            32,
            _model_zoo_path,
            model_def="embedding_test_module.EdlEmbeddingModel",
            channel=None,
        )
        worker._stub = InProcessMaster(master)

        inputs_list = [
            {
                "f1": tf.constant([[0], [1], [2]], tf.int64),
                "f2": tf.constant([[2], [1], [0]], tf.int64),
            },
            {
                "f1": tf.constant([[3], [4], [3]], tf.int64),
                "f2": tf.constant([[2], [1], [0]], tf.int64),
            },
        ]
        labels_list = [[0, 1, 0], [1, 1, 0]]
        input_dim = 5
        embedding_dim = 16
        worker.set_model(model_inst)

        # initialize kv store
        for layer in model_inst.layers:
            if isinstance(layer, Embedding):
                name = layer.name
                keys = [Embedding.get_key([name, i]) for i in range(input_dim)]
                values = [
                    np.random.rand(embedding_dim).astype(np.float32)
                    for i in range(input_dim)
                ]
                kv_store.update(keys, values)

        with mock.patch.object(
            EmbeddingService, "lookup_embedding", kv_store.lookup
        ), mock.patch.object(
            EmbeddingService, "update_embedding", kv_store.update
        ):
            worker._init_embedding_layer()
            worker._run_model_call_before_training(inputs_list[0])

            # run training process without tf.function
            correct_grads = []
            correct_ids_list = []
            for features, labels in zip(inputs_list, labels_list):
                loss, grads = worker.training_process_eagerly(features, labels)
                correct_grads.append(grads)
                ids = {}
                for layer in worker._embedding_layers:
                    ids[layer.name] = layer.embedding_and_ids[0].batch_ids
                correct_ids_list.append(ids)
                worker._reset_embedding()

            # run training process with tf.function
            test_grads = []
            test_ids_list = []
            for features, labels in zip(inputs_list, labels_list):
                self.assertFalse(worker._train_eagerly)
                loss, grads = worker.training_process(features, labels)
                test_grads.append(grads)
                ids = {}
                for layer in worker._embedding_layers:
                    ids[layer.name] = copy.deepcopy(
                        layer.embedding_and_ids[0].batch_ids
                    )
                test_ids_list.append(ids)
                worker._reset_embedding()

        # compare the gradients
        for test_g, correct_g in zip(test_grads, correct_grads):
            for g1, g2 in zip(test_g, correct_g):
                if isinstance(g1, tf.IndexedSlices):
                    self.assertTrue(np.isclose(g1.values, g2.values).all())
                    self.assertTrue(np.isclose(g1.indices, g2.indices).all())
                else:
                    self.assertTrue(np.isclose(g1, g2).all())

        for test_ids, correct_ids in zip(correct_ids_list, test_ids_list):
            for layer_name in correct_ids.keys():
                self.assertTrue(
                    tf.equal(test_ids[layer_name], correct_ids[layer_name])
                    .numpy()
                    .all()
                )


if __name__ == "__main__":
    unittest.main()
