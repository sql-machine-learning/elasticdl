import unittest
import numpy as np
import tensorflow as tf
from ps import ParameterServer
import ps_client


class PSClientTestCase(unittest.TestCase):
    def setUp(self):
        self.model_vars = {'x1': np.array([0.0, 0.0], dtype='float32'),
                           'y1': np.array([1.0, 1.0], dtype='float32'),
                           'x2': np.array([1.0, 0.0], dtype='float32'),
                           'y2': np.array([0.0, 1.0], dtype='float32')}
        self.names = list(self.model_vars.keys())

        self.grad_vars = {'x1': np.array([0.2, 0.2], dtype='float32'),
                          'y1': np.array([0.8, 0.8], dtype='float32'),
                          'x2': np.array([0.8, 0.2], dtype='float32'),
                          'y2': np.array([0.2, 0.8], dtype='float32')}

        self.correct_val1 = {'x1': np.array([-0.02, -0.02], dtype='float32'),
                             'y1': np.array([0.92, 0.92], dtype='float32'),
                             'x2': np.array([0.92, -0.02], dtype='float32'),
                             'y2': np.array([-0.02, 0.92], dtype='float32')}

        self.correct_val2 = {'x1': np.array([-0.04, -0.04], dtype='float32'),
                             'y1': np.array([0.84, 0.84], dtype='float32'),
                             'x2': np.array([0.84, -0.04], dtype='float32'),
                             'y2': np.array([-0.04, 0.84], dtype='float32')}

    def test1(self):
        test_configs = [(1, ps_client.no_partition),
                        (2, ps_client.hash_partition)]
        for ps_size, func in test_configs:
            var_partitioned = func(self.model_vars, ps_size)
            ps_list = [ParameterServer(tf.train.GradientDescentOptimizer(0.1),
                                       var_partitioned[i])
                       for i in range(ps_size)]
            for ps in ps_list:
                ps.start()

            psc = ps_client.ParameterServerClient(ps_configs=ps_list,
                                                  partition_func=func)
            pull_result = psc.pull()
            psc.push(grads=self.grad_vars)
            base_step, pull_result = psc.pull(min_step=1)
            np.testing.assert_equal(base_step, 1)
            for name in self.names:
                np.testing.assert_array_almost_equal(
                    pull_result[name], self.correct_val1[name])
            psc.push(grads=self.grad_vars)
            base_step, pull_result = psc.pull(min_step=2)
            np.testing.assert_equal(base_step, 2)
            for name in self.names:
                np.testing.assert_array_almost_equal(
                    pull_result[name], self.correct_val2[name])
            np.testing.assert_equal(psc.get_min_base_step(), 2)
            for ps in ps_list:
                ps.join()


if __name__ == "__main__":
    unittest.main()
