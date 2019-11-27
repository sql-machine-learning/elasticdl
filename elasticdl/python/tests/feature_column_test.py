import unittest

import numpy as np
import tensorflow as tf

from elasticdl.python.elasticdl.feature_column import (
    feature_column as edl_feature_column,
)


def call_feature_columns(feature_columns, input):
    dense_features = tf.keras.layers.DenseFeatures(feature_columns)
    return dense_features(input)


def generate_mock_embedding_vectors(ids, dimension):
    return np.array(
        [np.array([id] * dimension, dtype=np.float32) for id in ids]
    )


class EmbeddingColumnTest(unittest.TestCase):
    def test_feature_column_call(self):
        dimension = 32

        item_id_embedding = edl_feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(
                "item_id", num_buckets=128
            ),
            dimension=dimension,
        )
        item_id_embedding.lookup_embedding = lambda unique_ids: generate_mock_embedding_vectors(
            unique_ids, item_id_embedding.dimension
        )

        output = call_feature_columns(
            [item_id_embedding], {"item_id": [1, 2, 3]}
        )
        self.assertTrue(
            (
                output.numpy()
                == generate_mock_embedding_vectors([1, 2, 3], dimension)
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
