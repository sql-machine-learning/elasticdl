import os
import unittest

import numpy as np
import tensorflow as tf

from elasticdl.python.common.constants import JobType
from elasticdl.python.common.model_helper import get_model_file, load_module
from elasticdl.python.master.servicer import MasterServicer
from elasticdl.python.tests.in_process_master import InProcessMaster
from elasticdl.python.worker.worker import Worker

_model_file = get_model_file(
    os.path.dirname(os.path.realpath(__file__)), "test_module.custom_model"
)
m = load_module(_model_file).__dict__


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


class IndexedSlicesTest(unittest.TestCase):
    def _create_master_and_worker(self):
        model_inst = custom_model()
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
            2,
            _model_file,
            model_def="test_module.custom_model",
            channel=None,
        )
        worker._model = model_inst
        worker._stub = InProcessMaster(master)

        return master, worker

    def test_indices_slices_correctness(self):
        master, worker = self._create_master_and_worker()

        values1 = tf.convert_to_tensor(
            np.array([[1, 2, 3], [10, 11, 12]], dtype=np.float32)
        )
        values2 = tf.convert_to_tensor(
            np.array([[7, 8, 9], [4, 5, 6]], dtype=np.float32)
        )

        indices1 = tf.convert_to_tensor([0, 3], dtype=tf.int64)
        indices2 = tf.convert_to_tensor([2, 0], dtype=tf.int64)

        grads1 = [
            tf.IndexedSlices(values1, indices1),
            tf.reshape(
                tf.convert_to_tensor(range(1, 7), dtype=tf.float32),
                shape=(6, 1),
            ),
        ]
        grads2 = [
            tf.IndexedSlices(values2, indices2),
            tf.reshape(
                tf.convert_to_tensor(range(13, 19), dtype=tf.float32),
                shape=(6, 1),
            ),
        ]

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

        worker._model_version = 0
        worker.report_gradient(grads1)
        worker.report_gradient(grads2)

        for i, j in zip(master._model.values(), expected_weights):
            self.assertTrue(np.all(i.numpy() - j < 0.0001))

    def test_wrong_indices(self):
        master, worker = self._create_master_and_worker()

        values = tf.zeros((2, 3))
        indices = tf.convert_to_tensor([0, 4], dtype=tf.int64)
        grads = [tf.IndexedSlices(values, indices), tf.zeros((6, 1))]
        err_msg = ".*wrong indices %d, out of range %d" % (4, 3)
        worker._model_version = 0
        with self.assertRaisesRegex(ValueError, err_msg):
            worker.report_gradient(grads)

    def test_wrong_shape(self):
        master, worker = self._create_master_and_worker()

        values = tf.zeros((2, 4))
        indices = tf.convert_to_tensor([0, 3], dtype=tf.int64)
        grads = [tf.IndexedSlices(values, indices), tf.zeros((6, 1))]
        err_msg = ".*incompatible indexed slice dimension %d, expected %d" % (
            4,
            3,
        )
        worker._model_version = 0
        with self.assertRaisesRegex(ValueError, err_msg):
            worker.report_gradient(grads)


if __name__ == "__main__":
    unittest.main()
