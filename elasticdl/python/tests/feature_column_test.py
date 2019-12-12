import unittest

import numpy as np
import tensorflow as tf

from elasticdl.python.elasticdl.feature_column import feature_column


def call_feature_columns(feature_columns, input):
    dense_features = tf.keras.layers.DenseFeatures(feature_columns)
    return dense_features(input)


def generate_vectors_with_one_hot_value(ids, dimension):
    if isinstance(ids, tf.Tensor):
        ids = ids.numpy()

    return np.array(
        [
            (np.arange(dimension) == (id % dimension)).astype(np.int)
            for id in ids
        ]
    )


def generate_vectors_fill_with_id_value(ids, dimension):
    if isinstance(ids, tf.Tensor):
        ids = ids.numpy()

    return np.array([np.full((dimension), id) for id in ids])


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
            generate_vectors_with_one_hot_value(
                unique_ids, item_id_embedding.dimension
            )
        )

        output = call_feature_columns(
            [item_id_embedding], {"item_id": [1, 2, 3]}
        )

        self.assertTrue(
            np.array_equal(
                output.numpy(),
                generate_vectors_with_one_hot_value([1, 2, 3], dimension),
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
            generate_vectors_with_one_hot_value(
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

    def test_embedding_column_gradients(self):
        dimension = 8

        inputs = {"item_id": [[1, 2], [10, 10], [6, 3], [10, 2]]}

        # unique_ids: 1, 2, 10, 6, 3
        expected_sum_grads = [1, 2, 3, 1, 1]
        expected_mean_grads = [1 / 2.0, 2 / 2.0, 3 / 2.0, 1 / 2.0, 1 / 2.0]
        expected_sqrtn_grads = [
            1 / np.sqrt(2.0),
            2 / np.sqrt(2.0),
            3 / np.sqrt(2.0),
            1 / np.sqrt(2.0),
            1 / np.sqrt(2.0),
        ]

        combiner_to_expected_grads = {
            "sum": expected_sum_grads,
            "mean": expected_mean_grads,
            "sqrtn": expected_sqrtn_grads,
        }

        for combiner, expected_grads in combiner_to_expected_grads.items():
            item_id_embedding = feature_column.embedding_column(
                tf.feature_column.categorical_column_with_identity(
                    "item_id", num_buckets=128
                ),
                dimension=dimension,
                combiner=combiner,
            )
            item_id_embedding.lookup_embedding = lambda unique_ids: (
                generate_vectors_fill_with_id_value(
                    unique_ids, item_id_embedding.dimension
                )
            )

            dense_features = tf.keras.layers.DenseFeatures([item_id_embedding])
            call_fns = [dense_features.call, tf.function(dense_features.call)]

            for call_fn in call_fns:
                with tf.GradientTape() as tape:
                    item_id_embedding.set_tape(tape)
                    output = call_fn(inputs)
                    batch_embedding = item_id_embedding.embedding_and_ids[
                        0
                    ].batch_embedding
                    grads = tape.gradient(output, batch_embedding)

                    item_id_embedding.reset()

                    grads = grads.numpy()
                    for i in range(5):
                        self.assertTrue(
                            np.isclose(
                                grads[i],
                                np.full(dimension, expected_grads[i]),
                            ).all()
                        )

    def test_embedding_column_gradients_with_weights(self):
        dimension = 8

        inputs = {
            "item_id": [[1, 2], [10, 10], [6, 3], [10, 2]],
            "frequency": [[0.3, 0.6], [0.1, 0.2], [0.8, 0.1], [0.9, 0.6]],
        }

        item_ids = tf.reshape(inputs["item_id"], shape=[-1])
        _, item_id_idx = tf.unique(item_ids)

        # unique_ids: 1, 2, 10, 6, 3
        expected_sum_grads = np.array(
            [
                np.full(dimension, value)
                for value in [0.3, 0.6, 0.1, 0.2, 0.8, 0.1, 0.9, 0.6]
            ]
        )
        expected_mean_grads = np.array(
            [
                np.full(dimension, value)
                for value in [
                    0.3 / 0.9,
                    0.6 / 0.9,
                    0.1 / 0.3,
                    0.2 / 0.3,
                    0.8 / 0.9,
                    0.1 / 0.9,
                    0.9 / 1.5,
                    0.6 / 1.5,
                ]
            ]
        )
        expected_sqrtn_grads = np.array(
            [
                np.full(dimension, value)
                for value in [
                    0.3 / np.sqrt(np.power(0.3, 2) + np.power(0.6, 2)),
                    0.6 / np.sqrt(np.power(0.3, 2) + np.power(0.6, 2)),
                    0.1 / np.sqrt(np.power(0.1, 2) + np.power(0.2, 2)),
                    0.2 / np.sqrt(np.power(0.1, 2) + np.power(0.2, 2)),
                    0.8 / np.sqrt(np.power(0.8, 2) + np.power(0.1, 2)),
                    0.1 / np.sqrt(np.power(0.8, 2) + np.power(0.1, 2)),
                    0.9 / np.sqrt(np.power(0.9, 2) + np.power(0.6, 2)),
                    0.6 / np.sqrt(np.power(0.9, 2) + np.power(0.6, 2)),
                ]
            ]
        )

        combiner_to_expected_grads = {
            "sum": expected_sum_grads,
            "mean": expected_mean_grads,
            "sqrtn": expected_sqrtn_grads,
        }

        for combiner, expected_grads in combiner_to_expected_grads.items():
            item_id_embedding = feature_column.embedding_column(
                tf.feature_column.weighted_categorical_column(
                    tf.feature_column.categorical_column_with_identity(
                        "item_id", num_buckets=128
                    ),
                    weight_feature_key="frequency",
                ),
                dimension=dimension,
                initializer=tf.initializers.identity,
                combiner=combiner,
            )
            item_id_embedding.lookup_embedding = lambda unique_ids: (
                generate_vectors_fill_with_id_value(
                    unique_ids, item_id_embedding.dimension
                )
            )

            dense_features = tf.keras.layers.DenseFeatures([item_id_embedding])
            call_fns = [dense_features.call, tf.function(dense_features.call)]

            for call_fn in call_fns:
                with tf.GradientTape() as tape:
                    item_id_embedding.set_tape(tape)
                    output = call_fn(inputs)
                    batch_embedding = item_id_embedding.embedding_and_ids[
                        0
                    ].batch_embedding
                    grads = tape.gradient(output, batch_embedding)

                    item_id_embedding.reset()

                    grad_indices = grads.indices
                    grad_values = grads.values

                    self.assertTrue(
                        np.array_equal(grad_indices.numpy(), item_id_idx.numpy())
                    )
                    self.assertTrue(
                        np.isclose(grad_values.numpy(), expected_grads).all()
                    )


if __name__ == "__main__":
    unittest.main()
