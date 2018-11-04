import tensorflow as tf
import numpy as np
import threading


def data_generator():
    # XXX if there are too may training samples, training process will fail, not sure why.
    for _ in range(100):
        x = np.random.rand()
        yield [x], [x * 2 + 1]


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

    # Forward all other methods. TODO(l.zou): could use a proxy to automate these
    def get_slot(self, *args, **kwargs):
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        return self._optimizer.variables(*args, **kwargs)

    def apply_gradients(self, *args, **kwargs):
        return self._optimizer.apply_gradients(*args, **kwargs)


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


class HijackRunHook(tf.train.SessionRunHook):
    def __init__(self, name, ps, optimizer):
        self._name = name
        self._ps = ps
        self._optimizer = optimizer

    def before_run(self, run_context):
        print(self._name, run_context.original_args)
        return tf.train.SessionRunArgs(fetches={'grad': self._optimizer._grad_op, 'step': tf.train.get_global_step()})

    def after_run(self, run_context, run_values):
        if run_context.stop_requested:
            return
        print(self._name, run_context.original_args)
        print(self._name, run_values)

        # XXX Need a general way to find gradients.
        grad = run_values.results['grad']
        self._ps.push(
            np.array([grad[0][0][0], grad[1][0][0]], dtype='float') * -0.1)

        if run_values.results['step'] % 2 == 0:
            w = self._ps.pull()
            print(self._name, "pull: ", w)
            # XXX need a general way to get variable names
            for v in tf.trainable_variables():
                if v.name.startswith('linear/linear_model/bias_weights/'):
                    bias = v
                elif v.name.startswith('linear/linear_model/x/weights/'):
                    x = v
            # XXX big hack
            run_context.session.graph._unsafe_unfinalize()
            run_context.session.run(
                [tf.assign(bias, [w[1]]), tf.assign(x, [[w[0]]])])


class Program(object):
    def __init__(self, name, ps):
        self._name = name
        self._ps = ps


    def input_fn(self):
        dataset = tf.data.Dataset.from_generator(
            generator=data_generator, output_types=(tf.float32, tf.float32)).batch(1)
        feature, label = dataset.make_one_shot_iterator().get_next()
        return {'x': feature}, label


    def train(self):
        x_col = tf.feature_column.numeric_column('x')

        # Hijack optimizer
        optimizer = HijackGradientsOptimizer(
            tf.train.GradientDescentOptimizer(0.1))
        es = tf.estimator.LinearRegressor([x_col], optimizer=optimizer)
        # Hijack run process
        es = es.train(self.input_fn, hooks=[HijackRunHook(self._name, self._ps, optimizer)])
        for v in es.get_variable_names():
            print("%s-%s: %s" % (self._name, v, es.get_variable_value(v)))


class Worker(threading.Thread):
    def __init__(self, name, ps, prog):
        self._prog = prog(name, ps)
        threading.Thread.__init__(self, name=name)

    def run(self):
        self._prog.train()


def main():
    ps = ParameterServer()

    worker1 = Worker('worker1', ps, Program)
    worker2 = Worker('worker2', ps, Program)

    worker1.start()
    worker2.start()

    worker1.join()
    worker2.join()

if __name__ == '__main__':
    main()
