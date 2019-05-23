import tensorflow as tf
import numpy as np

class Dummy(object):
    # This should be called first such that optimizer() and vars() would return valid values.
    def init(self):
        self.opt_ = tf.train.GradientDescentOptimizer(0.1)
        self.vars_ = {'x1': np.array([0.0, 0.0], dtype='float32'),
                     'y1': np.array([1.0, 1.0], dtype='float32')}

    def optimizer(self):
        return self.opt_

    def vars(self):
        return self.vars_

