import tensorflow as tf
tf.enable_eager_execution()
import tensorflow.contrib.eager as tfe
import numpy as np
import queue
import threading


class ParameterServer(object):
    def __init__(self, optimizer, vars):
        self._opt = optimizer
        self._vars = {}
        for k, v in vars.items():
            if (not isinstance(v, np.ndarray)
                    or v.dtype not in (np.float32, np.float64)):
                raise ValueError(
                    'Initial value for variable %s is not of float type ndarray' %
                    k)
            self._vars[k] = tfe.Variable(v, name=k)
        self._step = 0
        self._grad_q = queue.Queue()
        self._lock = threading.Lock()
        self._runner = threading.Thread(target=self._run, name='ps-runner')
        self._exiting = False

    def pull(self, names=None):
        with self._lock:
            if names:
                res = {k: self._vars[k].numpy() for k in names}
            else:
                res = {k: v.numpy() for k, v in self._vars.items()}
        return self._step, res

    def push(self, base_step, sub_step, grads):
        if base_step > self._step:
            raise ValueError(
                'Illegal base step %s, parameter server step is %s' %
                (base_step, self._step))
        if sub_step < 0:
            raise ValueError('Illegal sub step %s' % sub_step)

        for k, g in grads.items():
            v = self._vars[k]
            if g.dtype != v.dtype.as_numpy_dtype or g.shape != v.shape:
                raise ValueError('Incompatible gradient for variable %s' % k)
        # TODO(l.zou): use @dataclass when python 3.7 is available.
        self._grad_q.put((base_step, sub_step, grads))

    def _compute(self, grads):
        grads_vars = [(g, self._vars[k]) for k, g in grads.items()]
        with self._lock:
            self._opt.apply_gradients(grads_vars)
            self._step += 1

    def _run(self):
        while not self._exiting:
            # TODO(l.zou): How to properly accumulate and decay grads?
            try:
                base_step, sub_step, grads = self._grad_q.get(timeout=1.0)
                self._compute(grads)
            except queue.Empty:
                pass

    def start(self):
        self._runner.start()

    def join(self):
        self._exiting = True
        self._runner.join()
