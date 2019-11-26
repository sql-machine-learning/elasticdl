import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf

from elasticdl.python.ps.learning_rate_modulator import (
    add_lr_modulation_to_optimizer,
)


class LearningRateTest(unittest.TestCase):
    @staticmethod
    def get_lr(lr_modulation, opt, multiplier):
        lr_modulation.set_multiplier(multiplier)
        # sleep 1s to wait that all threads are in this method call
        time.sleep(1)
        return opt.learning_rate

    @staticmethod
    def apply_gradients_with_modulation(
        lr_modulation, opt, multiplier, variables, grads
    ):
        grads_and_vars = zip(grads, variables)
        lr_modulation.set_multiplier(multiplier)
        # sleep 1s to wait that all threads are in this method call
        time.sleep(1)
        opt.apply_gradients(grads_and_vars)
        return [v.numpy() for v in variables]

    def test_lr_modulation(self):
        lr = 0.1
        multipliers = [1, 0.5, 0.1, 0.01]
        counts = len(multipliers)
        opt = tf.optimizers.SGD(lr)
        lr_modulation = add_lr_modulation_to_optimizer(opt)

        executor = ThreadPoolExecutor(max_workers=counts)
        tasks = [
            executor.submit(self.get_lr, lr_modulation, opt, m)
            for m in multipliers
        ]
        results = [tasks[i].result() for i in range(counts)]
        for i in range(counts):
            self.assertAlmostEqual(results[i], lr * multipliers[i])

        variables = []
        grads = []
        original_values = [1.2, 0.8]
        grad_values = [0.2, 0.1]

        for i in range(counts):
            variables.append([tf.Variable(v) for v in original_values])
            grads.append([tf.convert_to_tensor(g) for g in grad_values])

        tasks = [
            executor.submit(
                self.apply_gradients_with_modulation,
                lr_modulation,
                opt,
                multipliers[i],
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
                    i_diff[j], grad_values[j] * lr * multipliers[i], place
                )


if __name__ == "__main__":
    unittest.main()
