import unittest

import numpy as np
import tensorflow as tf

from elasticdl.python.elasticdl.feature_column import feature_column


def call_feature_columns(feature_columns, input):
    dense_features = tf.keras.layers.DenseFeatures(feature_columns)
    return dense_features(input)


def generate_mock_embedding_vectors(ids, dimension):
    if isinstance(ids, tf.Tensor):
        ids = ids.numpy()

    return np.array(
        [
            (np.arange(dimension) == (id % dimension)).astype(np.int)
            for id in ids
        ]
    )


class EmbeddingColumnTest(unittest.TestCase):
    def test_call_embedding_column(self):
        dimension = 32

        item_id_embedding = feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(
                "item_id", num_buckets=128
            ),
            dimension=dimension,
        )
        item_id_embedding.lookup_embedding = lambda unique_ids: (
            generate_mock_embedding_vectors(
                unique_ids, item_id_embedding.dimension
            )
        )

        output = call_feature_columns(
            [item_id_embedding], {"item_id": [1, 2, 3]}
        )

        self.assertTrue(
            np.array_equal(
                output.numpy(),
                generate_mock_embedding_vectors([1, 2, 3], dimension),
            )
        )

    def test_call_embedding_column_with_weights(self):
        dimension = 8

        item_id_embedding = feature_column.embedding_column(
            tf.feature_column.weighted_categorical_column(
                tf.feature_column.categorical_column_with_identity(
                    "item_id", num_buckets=128
                ),
                weight_feature_key="frequency",
            ),
            dimension=dimension,
            initializer=tf.initializers.identity,
            combiner="sum",
        )
        item_id_embedding.lookup_embedding = lambda unique_ids: (
            generate_mock_embedding_vectors(
                unique_ids, item_id_embedding.dimension
            )
        )

        output = call_feature_columns(
            [item_id_embedding],
            {
                "item_id": [[2, 6, 5], [3, 1, 1]],
                "frequency": [[0.33, 5.0, 1.024], [2.048, 0.5, 1.0]],
            },
        )

        expected_output = np.array(
            [
                [0.0, 0.0, 0.33, 0.0, 0.0, 1.024, 5.0, 0.0],
                [0.0, 1.5, 0.0, 2.048, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        self.assertTrue(np.array_equal(output.numpy(), expected_output))


if __name__ == "__main__":
    unittest.main()
