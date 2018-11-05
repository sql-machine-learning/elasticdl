import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import threading


# The following is supposed to be the content of a user program.

# ElasticFlow users write a deep learning training program that can
# run locally and distributedly simply by define a module which includes
# (1). a forward function to define the network;
# (2). a loss function for the loss;
# (3). a optimizer function for the optimizer.
# (4). For eager execution only, get_trainable_var_list returns the trainable variable list.
class UserDefinedModule(object):
    def __init__(self):
        self._trainable_var_list = []
        rand_value = np.random.rand(2)
        self._W = tfe.Variable(rand_value[0], name = "W", dtype = tf.float32)
        self._trainable_var_list.append(self._W)
        self._b = tfe.Variable(rand_value[1], name = "b", dtype = tf.float32)
        self._trainable_var_list.append(self._b)

    def forward(self, x):
        return self._W * x + self._b

    def loss(self, y_predict, y_true):
        return tf.square(y_true - y_predict)

    def optimizer(self):
        return tf.train.GradientDescentOptimizer(0.1)

    def get_trainable_var_list(self):
        return self._trainable_var_list


# The following is supposed to be part of the ElasticFlow framework.

class HijackGradientsOptimizer(tf.train.Optimizer):
    def __init__(self, optimizer):
        self._optimizer = optimizer
        # TODO(l.zou): Need to understand use_locking
        super(HijackGradientsOptimizer, self).__init__(
            name = "HijackGradientsOptimizer", use_locking = False)

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

    # TODO(haitao): we can optimize data communication(quantization, etc)
    # by modifying worker side compute_gradients and ps side apply_gradients.
    def apply_gradients(self, *args, **kwargs):
        return self._optimizer.apply_gradients(*args, **kwargs)


class DataSource(object):
    @staticmethod
    def gen():
        for i in range(100):
            x = np.random.rand()
            yield(x, 2 * x + 1)

    def __init__(self):
        self._dataset = tf.data.Dataset.from_generator(self.gen,
                                    (tf.float32, tf.float32),
                                    (tf.TensorShape([]), tf.TensorShape([])))

    def get_data(self):
        return self._dataset


class ParameterServer(object):
    def __init__(self, module_cls):
        self._lock = threading.Lock()
        self._model = module_cls()
        self._opt = HijackGradientsOptimizer(self._model.optimizer())
        trainable_var_list = self._model.get_trainable_var_list()
        self._vars = {}
        for v in trainable_var_list:
            self._vars[v.name] = v
        
    def push(self, grad):
        with self._lock:
            assert len(self._vars) >= len(grad)
            grad_and_vars = []
            for v_name in grad:
                grad_and_vars.append((grad[v_name], self._vars[v_name]))
            self._opt.apply_gradients(grad_and_vars)

    def pull(self):
        with self._lock:
            vars_copy = {}
            for v_name in self._vars:
                vars_copy[v_name] = self._vars[v_name].numpy()
            return vars_copy


class Worker(threading.Thread):
    def __init__(self, name, ds, ps, module_cls):
        self._name = name
        self._ds = ds
        self._ps = ps
        self._model = module_cls()
        self._trainable_var_list = self._model.get_trainable_var_list()
        self._vars = {}
        for v in self._trainable_var_list:
            self._vars[v.name] = v
        self._opt = HijackGradientsOptimizer(self._model.optimizer())
        threading.Thread.__init__(self, name=name)

    def update_param(self, vals):
        assert len(self._vars) == len(vals)
        for v_name in vals:
            tf.assign(self._vars[v_name], vals[v_name])

    def run(self):
        def closure():
            dataset = self._ds.get_data()
            for v in dataset:
                w = self._ps.pull()
                self.update_param(w)
                print("[%s] pull and update weight: "%self._name, w.items())

                # Forward and backward pass
                with tf.GradientTape() as tape:
                    predict = self._model.forward(v[0])
                    loss = self._model.loss(predict, v[1]) 
                grads = tape.gradient(loss, self._trainable_var_list)

                # Collect local gradients
                local_grad = {}
                for idx,  v in enumerate(self._trainable_var_list):
                    local_grad[v.name] = grads[idx].numpy()
                self._ps.push(local_grad)
                print("[%s] push grad: "%self._name, local_grad.items())

        t = threading.Thread(target = closure, name = self._name)
        t.start()
        return t


def main():
    # Enable eager execution
    tf.enable_eager_execution()

    ps = ParameterServer(UserDefinedModule)

    worker1 = Worker('worker1', DataSource(), ps, UserDefinedModule)
    worker2 = Worker('worker2', DataSource(), ps, UserDefinedModule)

    worker1.start()
    worker2.start()

    worker1.join()
    worker2.join()


if __name__ == '__main__':
    main()
