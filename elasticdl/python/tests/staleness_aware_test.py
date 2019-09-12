import unittest
import time

import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

from elasticdl.python.master.lr_modulation import (
    add_lr_modulation_to_optimizer,
)


class LearningRateTest(unittest.TestCase):
    @staticmethod
    def get_lr(lr_modulation, opt, multiplier):
        lr_modulation.set_multiplier(multiplier)
        time.sleep(1)
        return opt.learning_rate

    @staticmethod
    def apply_gradients_with_modulation(
        lr_modulation, opt, multiplier, variables, grads
    ):
        grads_and_vars = zip(grads, variables)
        lr_modulation.set_multiplier(multiplier)
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
        first_diff = [
            results[0][i] - original_values[i]
            for i in range(len(original_values))
        ]
        place = 5
        for i in range(1, counts):
            i_diff = [
                results[i][j] - original_values[j]
                for j in range(len(original_values))
            ]
            for j in range(len(original_values)):
                # variable value change ratio equals the learning rate ratio
                # for SGD without momentum
                self.assertAlmostEqual(
                    i_diff[j] / first_diff[j],
                    multipliers[i] / multipliers[0],
                    place,
                )


if __name__ == "__main__":
    unittest.main()
