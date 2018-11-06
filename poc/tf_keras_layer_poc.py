import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import variables
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 
import itertools
import threading
import os
import time

print(tf.__version__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class DataSource(object):
    def __init__(self, batch_size):
        self._batch_size = batch_size  
        self._mnist = input_data.read_data_sets("/tmp/elastic_flow_data", one_hot=True, reshape=False)

    def get_data(self):
        ''' real data for mnist images '''
        return self._mnist.train.next_batch(self._batch_size)

    def validation_inputs(self):
        return self._mnist.validation.images 

    def validation_labels(self):
        return self._mnist.validation.labels

class ParameterServer(object):
    def __init__(self, name, user_defined_module, learning_rate):
        self._name = name
        self._lock = threading.Lock()
        self._udm = user_defined_module(name)

        ''' initialize all the variables in ps '''
        with self._udm._graph.as_default():
            module_loss = self._udm.loss()
            self._optimizer = HijackGradientsOptimizer(self._udm.optimizer(learning_rate))
            self._optimizer.minimize(module_loss, global_step=tf.train.get_global_step())
            self._udm._sess.run(tf.global_variables_initializer())
  
    def push(self, gradients):
        assert len(tf.trainable_variables()) == len(gradients)
        with self._lock:
            with self._udm._graph.as_default():
                grad_and_vars = zip(gradients, tf.trainable_variables()) 
                self._udm._sess.run(self._optimizer.apply_gradients(grad_and_vars))
    
    def pull(self):
        with self._lock:
            with self._udm._graph.as_default():
                vars_cp = []
                for index in range(len(tf.trainable_variables())):
                    var = tf.trainable_variables()[index]
                    val = var.eval(session=self._udm._sess)
                    vars_cp.append(val)
                return vars_cp 

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

    def apply_gradients(self, *args, **kwargs):
        return self._optimizer.apply_gradients(*args, **kwargs)

class Worker(object):
    def __init__(self, name, ps, ds, user_module, learning_rate, batch_size):
        self._name = name
        self._ps = ps
        self._ds = ds
        self._user_module = user_module(name)
        self._learning_rate = learning_rate
        self._batch_size = batch_size

    def run(self):
        def closure():
            with self._user_module._graph.as_default(), self._user_module._sess as sess:

                module_loss = self._user_module.loss()
                optimizer = HijackGradientsOptimizer(self._user_module.optimizer(0.1))
                train_op = optimizer.compute_gradients(module_loss)
                grad_op = optimizer._grad_op

                sess.run(tf.global_variables_initializer())
                 
                for i in range(1000):
                    if True:
                        ps_var_values = self._ps.pull()
                        if i % 10 == 0:
                            print("pull: ", self._name, 'total get ' + str(len(ps_var_values)) + ' variables from ps')
 
                        trainable_vars = tf.trainable_variables()
                        for index in range(len(trainable_vars)):
                            trainable_var = trainable_vars[index]
                            ps_var_value = ps_var_values[index]
                            sess.run(tf.assign(trainable_var, ps_var_value))
                                 
                    batch_xs, batch_ys = self._ds.get_data()

                    _, g = self._user_module._sess.run([train_op, grad_op], feed_dict = 
                        {self._user_module.inputs():batch_xs, self._user_module.labels():batch_ys})
                    self._user_module.forward(module_loss, i, batch_xs, batch_ys, self._ds.validation_inputs(), self._ds.validation_labels())

                    gradients = []
                    for index in range(len(g)):
                        item = g[index]
                        grad = item[0]
                        gradients.append(grad)
                    self._ps.push(gradients)
                    if i % 10 == 0:
                        print("push: ", self._name, 'total upload ' + str(len(g)) + ' variables from worker')

        t = threading.Thread(target = closure, name = self._name)
        t.start()
        return t

class UserDefinedModule(object):
    ''' base class for define the net '''
    def __init__(self, name, learning_rate=None):
        pass

    def forward(self, batch_i, batch_x, batch_y, validation_inputs, validation_labels):
        pass

    def loss(self, x, y):
        pass

    def optimizer(self):
        pass

    def inputs(self):
        pass

    def labels(self):
        pass

class KerasCNN(UserDefinedModule):
    def __init__(self, name):
        ''' A simple CNN '''
        self._name = name
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._inputs = tf.placeholder(tf.float32, [None, 28, 28, 1])
            self._labels = tf.placeholder(tf.float32, [None, 10])

            conv1 = tf.layers.conv2d(self._inputs, 32, 5, activation=tf.nn.relu)
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
            conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
            fc = tf.contrib.layers.flatten(conv2)
            fc = tf.layers.dense(fc, 1024)
            fc = tf.layers.dropout(fc, rate=0.75, training=True)
            self._logits = tf.layers.dense(fc, 10)

            self._correct_prediction = tf.equal(tf.argmax(self._logits, 1), tf.argmax(self._labels, 1))
            self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction, tf.float32))

            self._sess = tf.Session()

    def forward(self, loss_t, batch_i, batch_x, batch_y, validation_inputs, validation_labels):
        if batch_i % 10 == 0:
            loss, acc = self._sess.run([loss_t, self._accuracy], 
                {self._inputs: validation_inputs, self._labels: validation_labels})
            print('{:>s} Batch: {:>2}: Validation loss: {:>3.5f}, Validation accuracy: {:>3.5f}'.format(self._name, batch_i, loss, acc))

    def loss(self):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self._logits, labels=self._labels)) 

    def optimizer(self, learning_rate):
        return tf.train.AdamOptimizer(learning_rate=learning_rate)

    def inputs(self):
        return self._inputs

    def labels(self):
        return self._labels

def main():
    learning_rate = 0.01
    batch_size = 128

    ps = ParameterServer('ps', KerasCNN, learning_rate * 0.5)
    worker1 = Worker("worker1", ps, DataSource(batch_size), KerasCNN, learning_rate, batch_size).run()
    worker2 = Worker("worker2", ps, DataSource(batch_size), KerasCNN, learning_rate, batch_size).run()
    worker1.join()
    worker2.join()

if __name__ == '__main__':
    main()
