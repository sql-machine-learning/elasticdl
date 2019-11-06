import os
import unittest

import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.model_utils import get_model_spec
from elasticdl.python.worker.worker import Worker


def random_batch(batch_size):
    shape = (28, 28)
    shape = (batch_size,) + shape

    num_classes = 10

    images = tf.random.uniform(shape)
    labels = tf.random.uniform(
        [batch_size], minval=0, maxval=num_classes, dtype=tf.int32
    )

    return images, labels


_model_zoo_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../../model_zoo"
)


class WorkerMnistTest(unittest.TestCase):
    def test_train(self):

        model_def = "mnist_functional_api.mnist_functional_api.custom_model"
        batch_size = 16

        worker = Worker(
            worker_id=0,
            job_type=elasticdl_pb2.TRAINING,
            minibatch_size=batch_size,
            model_zoo=_model_zoo_path,
            model_def=model_def,
            ps_channels=None,
        )

        print(worker)

        (
            model_inst,
            dataset_fn,
            loss_fn,
            opt_fn,
            eval_metrics_fn,
            prediction_outputs_processor,
        ) = get_model_spec(
            model_zoo=_model_zoo_path,
            model_def=model_def,
            dataset_fn="dataset_fn",
            loss="loss",
            optimizer="optimizer",
            eval_metrics_fn="eval_metrics_fn",
            prediction_outputs_processor="PredictionOutputsProcessor",
        )

        model = model_inst
        images, labels = random_batch(batch_size)
        with tf.GradientTape() as tape:
            output = model.call(images, training=True)
            labels = tf.reshape(labels, [-1])
            loss = loss_fn(output, labels)
            grads = tape.gradient(loss, model.trainable_variables)
        print(grads)


if __name__ == "__main__":
    unittest.main()
