import os
import tempfile
import unittest
from contextlib import closing

import mock
import numpy as np
import recordio
import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.constants import JobType
from elasticdl.python.master.embedding_service import EmbeddingService
from elasticdl.python.master.evaluation_service import EvaluationService
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.master.task_dispatcher import _TaskDispatcher
from elasticdl.python.tests.in_process_master import InProcessMaster
from elasticdl.python.worker.worker import Worker

_model_zoo_path = os.path.dirname(os.path.realpath(__file__))


# TODO (yunjian.lmh): Remove MockEmbeddingService, use MockKvStore instead
class MockEmbeddingService:
    def __init__(self):
        self.mock_embedding_table = None

    def mock_lookup_embedding(self, **kwargs):
        keys = kwargs["keys"]
        embeddings = []
        unknown_index = []
        for index, k in enumerate(keys):
            if k in self.mock_embedding_table:
                embeddings.append(
                    self.mock_embedding_table[k].reshape((1, -1))
                )
            else:
                unknown_index.append(index)
                embeddings.append(None)
        return embeddings, unknown_index

    def mock_update_embedding(self, **kwargs):
        keys, embeddings = kwargs["keys"], kwargs["embedding_vectors"]
        if embeddings is None:
            return
        for k, emb in zip(keys, embeddings):
            self.mock_embedding_table[k] = emb


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

        shards = {create_recordio_file(128): 128}
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

    def test_lookup_embedding(self):
        mock_embedding_service = MockEmbeddingService()

        ids = [1, 2, 3, 4, 5, 6]
        layer_name = "test_edlembedding"
        embedding_table_dim = 10
        mock_embedding_service.mock_embedding_table = {
            "test_edlembedding-1": np.zeros(
                (1, embedding_table_dim), dtype=np.float32
            ),
            "test_edlembedding-2": np.zeros(
                (1, embedding_table_dim), dtype=np.float32
            ),
            "test_edlembedding-3": np.zeros(
                (1, embedding_table_dim), dtype=np.float32
            ),
        }
        worker = Worker(
            1,
            JobType.TRAINING_ONLY,
            32,
            _model_zoo_path,
            model_def="embedding_test_module.EdlEmbeddingModel",
            channel=None,
        )
        with mock.patch.object(
            EmbeddingService,
            "lookup_embedding",
            mock_embedding_service.mock_lookup_embedding,
        ), mock.patch.object(
            EmbeddingService,
            "update_embedding",
            mock_embedding_service.mock_update_embedding,
        ):
            e_lookup, e_unknown = EmbeddingService.lookup_embedding(
                keys=["-".join([layer_name, str(id)]) for id in ids]
            )
            lookup_result = worker.lookup_embedding(
                ids=ids,
                layer_name=layer_name,
                embedding_table_dim=embedding_table_dim,
            )
            self.assertTrue(len(e_lookup) == 6)
            self.assertTrue(len(e_unknown) == 3)
            self.assertTrue(len(lookup_result) == 6)
            self.assertFalse(None in lookup_result)


if __name__ == "__main__":
    unittest.main()
