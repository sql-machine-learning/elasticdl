import unittest

import tensorflow as tf

from elasticdl.python.data.parallel_transform import ParallelTransform


class ParallelTransformTest(unittest.TestCase):
    def test_transform_int(self):
        def transform(record):
            return record + 1

        records = [k for k in range(100)]
        pt = ParallelTransform(
            records=records, num_parallel_processes=2, transform_fn=transform
        )
        results = []
        while True:
            data = pt.next_data()
            if data is None:
                break
            results.append(data)

        expect_results = [k + 1 for k in range(100)]
        self.assertListEqual(results, expect_results)

    def test_create_tf_dataset(self):
        def transform(record):
            return record + 1

        def gen():
            records = [k for k in range(100)]
            pt = ParallelTransform(
                records=records,
                num_parallel_processes=4,
                transform_fn=transform,
            )
            while True:
                data = pt.next_data()
                if data is None:
                    break
                yield data

        ds = tf.data.Dataset.from_generator(gen, tf.int64)
        ds = ds.batch(10).prefetch(1)
        results = []
        for data in ds:
            results.extend(data.numpy().tolist())

        expect_results = [k + 1 for k in range(100)]
        self.assertListEqual(results, expect_results)


if __name__ == "__main__":
    unittest.main()
