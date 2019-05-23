import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import threading


# The following is supposed to be the content of a user program.

# ElasticFlow users write a deep learning training program that can
# run locally and distributedly simply by define a module which includes
# (1). a forward function to define the network;
# (2). a loss function for the loss;
# (3). a optimizer function for the optimizer.
class UserDefinedModule(object):
    def __init__(self):
        input1 = tf.keras.layers.Input(shape=(1,))
        x1 = tf.keras.layers.Dense(1)(input1)
        self._model = tf.keras.models.Model(input1, x1)

    def forward(self, x):
        return self._model.call(x)

    def loss(self, x, y):
        return tf.reduce_mean(tf.square(self.forward(x) - y))

    def optimizer(self):
        return tf.train.GradientDescentOptimizer(0.1)


# The following is supposed to be part of the ElasticFlow framework.

class DataSource(object):
    @staticmethod
    def gen():
        for _ in range(400):
            x = np.random.rand()
            yield([x], [2 * x + 1])

    def __init__(self):
        self._dataset = tf.data.Dataset.from_generator(
            self.gen, (tf.float32, tf.float32),
            (tf.TensorShape([1]), tf.TensorShape([1])))

    def get_data(self):
        return self._dataset

class ParameterServer(object):
    def __init__(self, module_cls):
        self._lock = threading.Lock()
        self._user_module = module_cls()
        self._opt = self._user_module.optimizer()
        
    def push(self, grad):
        with self._lock:
            grad_and_vars = []
            trainable_w = self._user_module._model.trainable_weights
            assert len(trainable_w) == len(grad)
            for idx, w in enumerate(trainable_w):
                grad_and_vars.append((grad[idx], w))
            self._opt.apply_gradients(grad_and_vars, global_step=tf.train.get_or_create_global_step())

    def pull(self):
        with self._lock:
            trainable_w = self._user_module._model.trainable_weights
            vars_copy = []
            for w in trainable_w:
                vars_copy.append(w.numpy())
            return vars_copy


class Worker(threading.Thread):
    def __init__(self, name, ds, ps, module_cls):
        self._name = name
        self._ds = ds
        self._ps = ps
        self._user_module = module_cls()
        threading.Thread.__init__(self, name=name)

    def update_param(self, vals):
        trainable_w = self._user_module._model.trainable_weights
        assert len(trainable_w) == len(vals)
        for idx, v in enumerate(vals):
            tf.assign(trainable_w[idx], v)

    def run(self):
        def closure():
            batch_size = 2
            dataset = self._ds.get_data().batch(batch_size)
            for v in dataset:
                w = self._ps.pull()
                self.update_param(w)
                print("[%s] pull and update weight: " % self._name, w)

                # Forward and backward pass
                with tf.GradientTape() as tape:
                    loss = self._user_module.loss(v[0], v[1])
                grads = tape.gradient(loss, self._user_module._model.trainable_weights)

                # Collect local gradients
                local_grad = []
                for g in grads:
                    local_grad.append(g.numpy())
                self._ps.push(local_grad)
                print("[%s] push grad: " % self._name, local_grad)

        t = threading.Thread(target=closure, name=self._name)
        t.start()
        return t


def main():
    ps = ParameterServer(UserDefinedModule)

    worker1 = Worker('worker1', DataSource(), ps, UserDefinedModule)
    worker2 = Worker('worker2', DataSource(), ps, UserDefinedModule)

    worker1.start()
    worker2.start()

    worker1.join()
    worker2.join()


if __name__ == '__main__':
    main()
