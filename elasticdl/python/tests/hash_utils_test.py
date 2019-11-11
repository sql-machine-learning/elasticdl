import unittest

import numpy as np

from elasticdl.python.common.hash_utils import (
    int_to_id,
    scatter_embedding_vector,
)


class HashUtilTest(unittest.TestCase):
    def test_scatter_embedding_vector(self):
        vectors = np.array([[1, 2], [3, 4], [5, 6], [1, 7], [3, 9]])
        indices = np.array([0, 1, 2, 3, 4])
        num = 2

        expected_results = {}
        for i, item_id in enumerate(indices):
            ps_id = int_to_id(item_id, num)
            if ps_id not in expected_results:
                item_list = [item_id]
                expected_results[ps_id] = [
                    np.expand_dims(vectors[i, :], axis=0),
                    item_list,
                ]
            else:
                expected_results[ps_id][0] = np.concatenate(
                    (
                        expected_results[ps_id][0],
                        np.expand_dims(vectors[i, :], axis=0),
                    ),
                    axis=0,
                )
                expected_results[ps_id][1].append(item_id)
        results = scatter_embedding_vector(vectors, indices, num)

        for ps_id in range(num):
            np.testing.assert_array_equal(
                results[ps_id][0], expected_results[ps_id][0]
            )
            self.assertListEqual(results[ps_id][1], expected_results[ps_id][1])


if __name__ == "__main__":
    unittest.main()
