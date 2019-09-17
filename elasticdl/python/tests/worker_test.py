import copy
import os
import tempfile
import unittest
from contextlib import closing

import mock
import numpy as np
import recordio
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Flatten

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.constants import JobType
from elasticdl.python.elasticdl.layers.embedding import Embedding
from elasticdl.python.master.embedding_service import EmbeddingService
from elasticdl.python.master.evaluation_service import EvaluationService
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.master.task_dispatcher import _TaskDispatcher
from elasticdl.python.tests.in_process_master import InProcessMaster
from elasticdl.python.tests.mock_kv_store import MockKvStore
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


def create_recordio_file(size):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with closing(recordio.Writer(temp_file.name)) as f:
        for _ in range(size):
            x = np.random.rand(1).astype(np.float32)
            y = 2 * x + 1
            example_dict = {
                "x": tf.train.Feature(float_list=tf.train.FloatList(value=x)),
                "y": tf.train.Feature(float_list=tf.train.FloatList(value=y)),
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=example_dict)
            )
            f.write(example.SerializeToString())
    return temp_file.name


class WorkerTest(unittest.TestCase):
    def distributed_train_and_evaluate(self, training=True):
        """
        Run distributed training and evaluation with a local master.
        grpc calls are mocked by local master call.
        """

        class _Master(InProcessMaster):
            def ReportGradient(self, req):
                if 2 < self._m._version < 80:
                    # For testing of retrain when gradient not accepted.
                    # Increase master version to reject the gradient.
                    self._m._version += 1
                return self._m.ReportGradient(req, None)

            def ReportEvaluationMetrics(self, req):
                if 2 < self._m._version < 80:
                    # Testing of evaluation retries. Increase the master
                    # version so the evaluation metrics will not be accepted.
                    self._m._version += 1
                return self._m.ReportEvaluationMetrics(req, None)

        job_type = (
            JobType.TRAINING_ONLY if training else JobType.EVALUATION_ONLY
        )
        batch_size = 16
        worker = Worker(
            1,
            job_type,
            batch_size,
            _model_zoo_path,
            model_def="test_module.custom_model",
            channel=None,
        )

        shards = {create_recordio_file(128): (0, 128)}
        if training:
            training_shards = shards
            evaluation_shards = {}
        else:
            training_shards = {}
            evaluation_shards = shards
        task_d = _TaskDispatcher(
            training_shards,
            evaluation_shards,
            {},
            records_per_task=64,
            num_epochs=1,
        )
        if not training:
            evaluation_service = EvaluationService(
                None, None, task_d, 0, 0, 0, True
            )
            task_d.set_evaluation_service(evaluation_service)
        else:
            evaluation_service = None
        master = MasterServicer(
            2,
            batch_size,
            worker._opt_fn(),
            task_d,
            init_var=[],
            checkpoint_filename_for_init="",
            checkpoint_service=None,
            evaluation_service=evaluation_service,
        )
        worker._stub = _Master(master)

        for var in worker._model.trainable_variables:
            master.set_model_var(var.name, var.numpy())

        worker.run()

        req = elasticdl_pb2.GetTaskRequest()
        req.worker_id = 1
        task = master.GetTask(req, None)
        # No more task.
        self.assertTrue(not task.shard_name)

    def test_distributed_train_tf_example(self):
        self.distributed_train_and_evaluate(training=True)

    def test_distributed_evaluate_tf_example(self):
        self.distributed_train_and_evaluate(training=False)

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
                    self.assertTrue(
                        (g1.values - g2.values < 0.0001).numpy().all()
                    )
                    self.assertTrue(
                        tf.equal(g1.indices, g2.indices).numpy().all()
                    )
                else:
                    self.assertTrue((g1 - g2 < 0.0001).numpy().all())

        for test_ids, correct_ids in zip(correct_ids_list, test_ids_list):
            for layer_name in correct_ids.keys():
                self.assertTrue(
                    tf.equal(test_ids[layer_name], correct_ids[layer_name])
                    .numpy()
                    .all()
                )


if __name__ == "__main__":
    unittest.main()
