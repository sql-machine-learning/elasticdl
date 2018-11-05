import tensorflow as tf
import numpy as np
import itertools
import threading


class DataSource(object):
    def get_data(self):
        x = np.random.rand()
        return np.array([x, x * 2 + 1], dtype='float')


class ParameterServer(object):
    def __init__(self):
        self._w = np.array([0, 0], dtype='float')
        self._lock = threading.Lock()

    def push(self, grad):
        with self._lock:
            self._w += grad

    def pull(self):
        with self._lock:
            return self._w.copy()


class HijackGradientsOptimizer(tf.train.Optimizer):
    def __init__(self, optimizer):
        self._optimizer = optimizer
        # TODO(l.zou): Need to understand use_locking
        super(HijackGradientsOptimizer, self).__init__(
            name="HijackGradientsOptimizer", use_locking=False)

    # The method we want to intercept
    def compute_gradients(self, *args, **kwargs):
        self._grad_op = self._optimizer.compute_gradients(*args, **kwargs)
        return self._grad_op

    # Forward all other methods. TODO(l.zou): could use a proxy to automate
    # these
    def get_slot(self, *args, **kwargs):
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        return self._optimizer.variables(*args, **kwargs)

    def apply_gradients(self, *args, **kwargs):
        return self._optimizer.apply_gradients(*args, **kwargs)


class Worker(object):
    def __init__(self, name, ps, ds, prog):
        self._name = name
        self._ps = ps
        self._ds = ds
        self._prog = prog(name)

    def run(self):
        def closure():
            with self._prog._sess as sess:
                sess.run(tf.global_variables_initializer())
                for i in range(10):
                    if i % 2 == 0:
                        w = self._ps.pull()
                        print("pull: ", self._name, w)
                        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
                            vw = tf.get_variable('w')
                        sess.run(tf.assign(vw, w))
                        print("assign:", sess.run(vw))
                    g = self._prog.forward(*self._ds.get_data()) * -0.1
                    print("push: ", self._name, g)
                    self._ps.push(g)

        t = threading.Thread(target=closure, name=self._name)
        t.start()
        return t


class LinearRegression(object):
    def __init__(self, name):
        self._graph = tf.Graph()
        with self._graph.as_default(), tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # simple linear model
            self._x = tf.placeholder("float")
            self._y = tf.placeholder("float")
            w = tf.get_variable("w", [2, ])
            y_predict = tf.multiply(self._x, w[0]) + w[1]

            loss = tf.square(self._y - y_predict)

            optimizer = HijackGradientsOptimizer(
                tf.train.GradientDescentOptimizer(0.1))
            self._train_op = optimizer.minimize(loss)
            self._grad_op = optimizer._grad_op
            self._sess = tf.Session()

    def forward(self, x, y):
        print(x, y)
        _, grad = self._sess.run([self._train_op, self._grad_op], feed_dict={
                                 self._x: x, self._y: y})
        print(grad)
        return grad[0][0]


def main():
    ps = ParameterServer()
    worker1 = Worker("worker1", ps, DataSource(), LinearRegression).run()
    worker2 = Worker("worker2", ps, DataSource(), LinearRegression).run()
    worker1.join()
    worker2.join()


if __name__ == '__main__':
    main()
