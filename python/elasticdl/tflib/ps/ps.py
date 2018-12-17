import threading
import queue
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops


class ParameterServer(object):
    def __init__(self, optimizer_func, tf_vars):
        self._step = 0
        self._grad_q = queue.Queue()
        self._lock = threading.Lock()
        self._runner = threading.Thread(target=self._run, name="ps-runner")
        self._exiting = False
        self._min_step_cv = threading.Condition()

        self._grads_vars = {}

        graph = tf.Graph()
        with graph.as_default():
            for k, v in tf_vars.items():
                if not isinstance(v, np.ndarray) or v.dtype not in (
                    np.float32,
                    np.float64,
                ):
                    raise ValueError(
                        "Initial value for variable %s is not a float ndarray" %
                        k)
                # TODO: In graph mode we don't need to keep track of variables by
                # ourselves.
                self._grads_vars[k] = (
                    array_ops.placeholder(dtype=v.dtype),
                    tf.Variable(v, name=k),
                )
            optimizer = optimizer_func()
            optimizer_type = optimizer.pop("type", None)
            assert optimizer
            learning_rate_value = optimizer.pop("learning_rate", None)
            assert learning_rate_value
            self._lr_scale = 1.0
            self._lr_scale_placeholder = tf.placeholder(
                dtype=tf.float32, shape=[])
            self._lr = self._lr_scale_placeholder * learning_rate_value
            self._opt = optimizer_type(learning_rate=self._lr, **optimizer)
            self._apply_grad_op = self._opt.apply_gradients(
                self._grads_vars.values())
            init_op = tf.global_variables_initializer()

        self._sess = tf.Session(graph=graph)
        self._sess.run(init_op)

    def set_learning_rate_scale(self, scale):
        self._lr_scale = scale

    def pull(self, names=None, min_step=0, blocking=True, timeout=None):
        with self._min_step_cv:
            self._min_step_cv.wait_for(
                lambda: not blocking or min_step <= self._step, timeout=timeout
            )
        with self._lock:
            if min_step > self._step:
                raise LookupError(
                    "Required step is not ready yet: %s" %
                    min_step)
            if names:
                res = {
                    k: self._grads_vars[k][1].eval(
                        self._sess) for k in names}
            else:
                res = {k: v[1].eval(self._sess)
                       for k, v in self._grads_vars.items()}
            return self._step, res

    def push(self, base_step, sub_step, grads):
        with self._lock:
            if base_step > self._step:
                raise ValueError(
                    "Illegal base step %s, parameter server step is %s"
                    % (base_step, self._step)
                )

        if sub_step < 0:
            raise ValueError("Illegal sub step %s" % sub_step)

        for k, g in grads.items():
            v = self._grads_vars[k][1]
            if g.dtype != v.dtype.as_numpy_dtype or g.shape != v.shape:
                raise ValueError("Incompatible gradient for variable %s" % k)
        # TODO(l.zou): use @dataclass when python 3.7 is available.
        self._grad_q.put((base_step, sub_step, grads))

    def _compute(self, grads):
        with self._lock:
            feed_dict = {self._grads_vars[k][0]: v for k, v in grads.items()}
            feed_dict[self._lr_scale_placeholder] = self._lr_scale
            self._sess.run(self._apply_grad_op, feed_dict=feed_dict)
        with self._min_step_cv:
            self._step += 1
            self._min_step_cv.notify_all()

    def _run(self):
        while not self._exiting:
            # TODO(l.zou): How to properly accumulate and decay grads?
            try:
                _, _, grads = self._grad_q.get(timeout=1.0)
                self._compute(grads)
            except queue.Empty:
                pass

    def start(self):
        self._runner.start()

    def join(self):
        self._exiting = True
        self._runner.join()
        self._sess.close()
