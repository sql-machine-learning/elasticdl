import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf

from elasticdl.python.common.lr_scheduler import add_lr_scheduler_to_optimizer


def lr_scheduler_func(model_version):
    if model_version < 1:
        return 1
    elif model_version < 2:
        return 0.5
    else:
        return 0.1


class LearningRateTest(unittest.TestCase):
    @staticmethod
    def get_lr(lr_scheduler, opt, model_version):
        lr_scheduler.set_model_version(model_version)
        # sleep 1s to wait that all threads are in this method call
        time.sleep(1)
        return opt.learning_rate

    @staticmethod
    def apply_gradients_with_scheduler(
        lr_scheduler, opt, model_version, variables, grads
    ):
        grads_and_vars = zip(grads, variables)
        lr_scheduler.set_model_version(model_version)
        # sleep 1s to wait that all threads are in this method call
        time.sleep(1)
        opt.apply_gradients(grads_and_vars)
        return [v.numpy() for v in variables]

    def test_lr_scheduler(self):
        opt = tf.optimizers.SGD(0.1)
        lr_scheduler = add_lr_scheduler_to_optimizer(opt, lr_scheduler_func)

        model_versions = [0, 1, 2]
        counts = len(model_versions)
        executor = ThreadPoolExecutor(max_workers=counts)
        tasks = [
            executor.submit(self.get_lr, lr_scheduler, opt, v)
            for v in model_versions
        ]
        results = [tasks[i].result() for i in range(counts)]
        for i in range(counts):
            self.assertAlmostEqual(
                results[i], lr_scheduler_func(model_versions[i])
            )

        variables = []
        grads = []
        original_values = [1.2, 0.8]
        grad_values = [0.2, 0.1]

        for i in range(counts):
            variables.append([tf.Variable(v) for v in original_values])
            grads.append([tf.convert_to_tensor(g) for g in grad_values])

        tasks = [
            executor.submit(
                self.apply_gradients_with_scheduler,
                lr_scheduler,
                opt,
                model_versions[i],
                variables[i],
                grads[i],
            )
            for i in range(counts)
        ]
        results = [tasks[i].result() for i in range(counts)]
        place = 5
        for i in range(0, counts):
            i_diff = [
                original_values[j] - results[i][j]
                for j in range(len(original_values))
            ]
            for j in range(len(original_values)):
                # variable value change ratio equals the learning rate ratio
                # for SGD without momentum
                self.assertAlmostEqual(
                    i_diff[j],
                    grad_values[j] * lr_scheduler_func(model_versions[i]),
                    place,
                )


if __name__ == "__main__":
    unittest.main()
