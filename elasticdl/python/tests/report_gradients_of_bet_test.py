import os
import unittest

import mock
import numpy as np
import tensorflow as tf

from elasticdl.python.common.constants import JobType
from elasticdl.python.elasticdl.layers.embedding import EmbeddingAndIds
from elasticdl.python.master.embedding_service import EmbeddingService
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.tests.in_process_master import InProcessMaster
from elasticdl.python.worker.worker import Worker

_model_zoo_path = os.path.dirname(os.path.realpath(__file__))


# TODO (yunjian.lmh): Remove MockEmbeddingService, use MockKvStore instead
class MockEmbeddingService:
    def __init__(self):
        self.mock_embedding_table = None

    def mock_lookup_embedding(
        self, keys=[], embedding_service_endpoint=None, parse_type=np.float32
    ):
        embeddings = []
        for k in keys:
            layer_name, idx = k.split("-")
            idx = int(idx)
            embeddings.append(self.mock_embedding_table[layer_name][idx])
        return embeddings, []

    def mock_update_embedding(
        self,
        keys=[],
        embedding_vectors=[],
        embedding_service_endpoint=None,
        set_if_not_exist=False,
    ):
        if embedding_vectors is None:
            return
        for k, emb in zip(keys, embedding_vectors):
            layer_name, idx = k.split("-")
            idx = int(idx)
            self.mock_embedding_table[layer_name][idx] = emb


def custom_model():
    embedding_weight = np.array(range(1, 13), dtype=np.float32).reshape(4, 3)
    dense_weight = np.array(range(13, 19), dtype=np.float32).reshape(-1, 1)

    inputs = tf.keras.Input(shape=(2,), name="image")
    x = tf.keras.layers.Embedding(
        input_dim=4, output_dim=3, input_length=2, weights=[embedding_weight]
    )(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(1, use_bias=False, weights=[dense_weight])(
        x
    )
    return tf.keras.Model(inputs=inputs, outputs=outputs)


class MockEdlEmbedding:
    def __init__(self, name):
        self._name = name
        self.embedding_and_ids = []

    @property
    def name(self):
        return self._name


class ReportBETGradientTest(unittest.TestCase):
    def _create_master_and_worker(
        self, service_endpoint=None, embedding_dims={}
    ):
        model_inst = custom_model()
        master = MasterServicer(
            2,
            2,
            tf.optimizers.SGD(0.1),
            None,
            init_var=model_inst.trainable_variables,
            embedding_service_endpoint=service_endpoint,
            embedding_dims=embedding_dims,
            checkpoint_filename_for_init=None,
            checkpoint_service=None,
            evaluation_service=None,
        )
        worker = Worker(
            1,
            JobType.TRAINING_ONLY,
            2,
            _model_zoo_path,
            model_def="test_module.custom_model",
            channel=None,
        )
        worker.set_model(model_inst)
        worker._stub = InProcessMaster(master)

        return master, worker

    def test_report_bet_gradients_worker_to_master(self):
        layer_names = ["test_edlembedding_1", "test_edlembedding__2"]
        embedding_dims = dict([(layer, 3) for layer in layer_names])
        mock_embedding_service_endpoint = {"host": "1.1.1.1", "port": "12123"}
        master, worker = self._create_master_and_worker(
            mock_embedding_service_endpoint, embedding_dims
        )

        mock_embedding_service = MockEmbeddingService()
        mock_embedding_service.mock_embedding_table = dict(
            [
                (layer, np.zeros((5, 3), dtype=np.float32))
                for layer in layer_names
            ]
        )

        layer1 = MockEdlEmbedding(layer_names[0])
        layer1.embedding_and_ids = [
            EmbeddingAndIds(None, tf.constant([1, 2])),
            EmbeddingAndIds(None, tf.constant([2, 3])),
        ]

        layer2 = MockEdlEmbedding(layer_names[1])
        layer2.embedding_and_ids = [
            EmbeddingAndIds(None, tf.constant([3, 1])),
            EmbeddingAndIds(None, tf.constant([3, 4])),
        ]

        edlembed_grads = [
            tf.reshape(tf.range(1, 7, dtype=tf.float32), (2, 3)),
            tf.reshape(tf.range(7, 13, dtype=tf.float32), (2, 3)),
            tf.reshape(tf.range(13, 19, dtype=tf.float32), (2, 3)),
            tf.reshape(tf.range(19, 25, dtype=tf.float32), (2, 3)),
        ]

        worker._embedding_layers = [layer1, layer2]

        values1 = tf.convert_to_tensor(
            np.array([[1, 2, 3], [10, 11, 12]], dtype=np.float32)
        )
        indices1 = tf.convert_to_tensor([0, 3], dtype=tf.int64)

        grads1 = [
            tf.IndexedSlices(values1, indices1),
            tf.reshape(
                tf.convert_to_tensor(range(1, 7), dtype=tf.float32),
                shape=(6, 1),
            ),
        ]
        grads1.extend(edlembed_grads)

        worker._model_version = 0
        with mock.patch.object(
            EmbeddingService,
            "lookup_embedding",
            mock_embedding_service.mock_lookup_embedding,
        ), mock.patch.object(
            EmbeddingService,
            "update_embedding",
            mock_embedding_service.mock_update_embedding,
        ):
            worker.report_gradient(grads1)

        expected_edlembedding_grads = {
            layer1.name: tf.IndexedSlices(
                tf.concat(edlembed_grads[:2], axis=0),
                tf.concat(
                    [
                        layer1.embedding_and_ids[0].batch_ids,
                        layer1.embedding_and_ids[1].batch_ids,
                    ],
                    axis=0,
                ),
            ),
            layer2.name: tf.IndexedSlices(
                tf.concat(edlembed_grads[2:], axis=0),
                tf.concat(
                    [
                        layer2.embedding_and_ids[0].batch_ids,
                        layer2.embedding_and_ids[1].batch_ids,
                    ],
                    axis=0,
                ),
            ),
        }

        result = master._edl_embedding_gradients
        for name, grads in expected_edlembedding_grads.items():
            self.assertTrue(name in result)
            self.assertTrue(grads.indices.shape == result[name].indices.shape)
            self.assertTrue(grads.values.shape == result[name].values.shape)
            self.assertTrue(
                np.all(tf.equal(grads.indices, result[name].indices))
            )
            self.assertTrue(
                np.all(grads.values - result[name].values < 0.0001)
            )

        # make sure other gradients are calculated correctly
        values2 = tf.convert_to_tensor(
            np.array([[7, 8, 9], [4, 5, 6]], dtype=np.float32)
        )
        indices2 = tf.convert_to_tensor([2, 0], dtype=tf.int64)
        grads2 = [
            tf.IndexedSlices(values2, indices2),
            tf.reshape(
                tf.convert_to_tensor(range(13, 19), dtype=tf.float32),
                shape=(6, 1),
            ),
        ]
        grads2.extend(edlembed_grads)
        with mock.patch.object(
            EmbeddingService,
            "lookup_embedding",
            mock_embedding_service.mock_lookup_embedding,
        ), mock.patch.object(
            EmbeddingService,
            "update_embedding",
            mock_embedding_service.mock_update_embedding,
        ):
            worker.report_gradient(grads2)

        expected_weights = []
        expected_weights.append(
            np.array(
                [
                    [0.5, 1.3, 2.1],
                    [4.0, 5.0, 6.0],
                    [6.3, 7.2, 8.1],
                    [9.0, 9.9, 10.8],
                ]
            )
        )
        expected_weights.append(
            np.reshape(
                np.array([[12.3, 13.2, 14.1, 15.0, 15.9, 16.8]]), (6, 1)
            )
        )

        for i, j in zip(master._model.values(), expected_weights):
            self.assertTrue(np.all(i.numpy() - j < 0.0001))

    def test_report_bet_gradients_master_to_service(self):
        layer_names = ["test_layer_1", "test_layer_2"]
        embedding_dims = dict([(layer, 4) for layer in layer_names])
        mock_embedding_service_endpoint = {"host": "1.1.1.1", "port": "12123"}
        master, _ = self._create_master_and_worker(
            mock_embedding_service_endpoint, embedding_dims
        )

        mock_embedding_service = MockEmbeddingService()
        mock_embedding_service.mock_embedding_table = {
            layer_names[0]: np.zeros((2, 4), dtype=np.float32),
            layer_names[1]: np.zeros((4, 4), dtype=np.float32),
        }
        for i in range(2):
            mock_embedding_service.mock_embedding_table[layer_names[0]][
                i
            ].fill(i)
        for i in range(4):
            mock_embedding_service.mock_embedding_table[layer_names[1]][
                i
            ].fill(i)

        grads = tf.reshape(tf.range(28, dtype=tf.float32), (7, 4))
        indices = tf.convert_to_tensor([0, 1, 0, 2, 0, 2, 3])

        master._edl_embedding_gradients = {
            layer_names[0]: tf.IndexedSlices(grads[:3], indices[:3]),
            layer_names[1]: tf.IndexedSlices(grads[3:], indices[3:]),
        }

        with mock.patch.object(
            EmbeddingService,
            "lookup_embedding",
            mock_embedding_service.mock_lookup_embedding,
        ), mock.patch.object(
            EmbeddingService,
            "update_embedding",
            mock_embedding_service.mock_update_embedding,
        ):
            with master._lock:
                assert master._lock.locked()
                master._update_model()

        expected_embedding_table = {
            layer_names[0]: np.array(
                [[-0.8, -1, -1.2, -1.4], [0.6, 0.5, 0.4, 0.3]]
            ),
            layer_names[1]: np.array(
                [
                    [-1.6, -1.7, -1.8, -1.9],
                    [1, 1, 1, 1],
                    [-1.2, -1.4, -1.6, -1.8],
                    [0.6, 0.5, 0.4, 0.3],
                ]
            ),
        }

        for layer in layer_names:
            self.assertTrue(
                (
                    expected_embedding_table[layer]
                    - mock_embedding_service.mock_embedding_table[layer]
                    < 0.0001
                ).all()
            )

    def test_get_trainable_variable(self):
        master, worker = self._create_master_and_worker()
        layer = MockEdlEmbedding("test")
        layer.embedding_and_ids = [
            EmbeddingAndIds(tf.Variable([1, 2, 3], name="test_bet"), [1, 2, 3])
        ]
        worker._embedding_layers = [layer]
        train_vars = worker.get_trainable_items()
        self.assertTrue("embedding" in train_vars[0].name)
        self.assertTrue("dense" in train_vars[1].name)
        self.assertTrue("test_bet" in train_vars[2].name)


if __name__ == "__main__":
    unittest.main()
