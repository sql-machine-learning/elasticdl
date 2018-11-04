import tensorflow as tf
from tensorflow.python.ops import variables
import numpy as np 
import itertools
import threading

class DataSource(object):
    def __init__(self, batch_size):
        self._batch_size = batch_size  

    def get_data(self):
        ''' mock data for mnist images '''
        batch_xs = np.random.random((self._batch_size, 28, 28, 1))
        batch_yx = np.random.randint(10, size=(self._batch_size, 10))
        return batch_xs, batch_yx

class ParameterServer(object):
    def __init__(self):
        self._lock = threading.Lock()
        self._inited = False
        self._vars = []
  
    def init_vars(self, name, var_shapes):
        ''' Only one of the workers will initialize the ps with variables in neural network '''
        with self._lock:
            if(self._inited):
                pass
            else:
                self._inited = True
                for shape in var_shapes:
                    var = np.zeros(shape)
                    self._vars.append(var)
        
    def push(self, gradients):
        with self._lock:
            for index in range(len(gradients)):
                grad = gradients[index]
                var = self._vars[index]
                var += grad
    
    def pull(self):
        with self._lock:
            return self._vars

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
    def __init__(self, name, ps, ds, prog, learning_rate, batch_size):
        self._name = name
        self._ps = ps
        self._ds = ds
        self._prog = prog(name, learning_rate)
        self._learning_rate = learning_rate
        self._batch_size = batch_size

    def run(self):
        def closure():
            with self._prog._sess as sess:
                sess.run(tf.global_variables_initializer())

                ''' Initialize the ps with variables in neural network before running(only upload the shape of variable) '''
                var_shapes = []
                trainable_vars = tf.trainable_variables() 
                for var in trainable_vars:
                    var_shapes.append(var.shape)
                self._ps.init_vars(self._name, var_shapes)

                for i in range(10):
                    if i > 0 and i % 2 == 0:
                        ps_var_values = self._ps.pull()
                        print("pull: ", self._name, 'total get ' + str(len(ps_var_values)) + ' variables from ps')

                        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
                            for index in range(len(trainable_vars)):
                                trainable_var = trainable_vars[index]
                                ps_var_value = ps_var_values[index]
                                sess.run(tf.assign(trainable_var, ps_var_value))
                    batch_xs, batch_ys = self._ds.get_data()
                    g = self._prog.forward(batch_xs, batch_ys)

                    print("push: ", self._name, 'total upload ' + str(len(g)) + ' variables from worker')
                    gradients = []
                    for item in g:
                        grad = item[0]
                        grad = grad * -1 * self._learning_rate
                        gradients.append(grad)
                    self._ps.push(gradients)
        
        t = threading.Thread(target = closure, name = self._name)
        t.start()
        return t
    
class KerasCNN(object):
    def __init__(self, name, learning_rate):
        ''' A simple CNN '''
        self._graph = tf.Graph()
        with self._graph.as_default(), tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self._inputs = tf.placeholder(tf.float32, [None, 28, 28, 1])
            self._labels = tf.placeholder(tf.float32, [None, 10])
            self._learning_rate = learning_rate

            layer = self._inputs
            for layer_i in range(1, 20):
                layer = self.conv_layer(layer, layer_i)
        
            orig_shape = layer.get_shape().as_list()
            layer = tf.reshape(layer, shape=[-1, orig_shape[1] * orig_shape[2] * orig_shape[3]])
            layer = self.fully_connected(layer, 100)
            logits = tf.layers.dense(layer, 10)

            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self._labels))
            optimizer = HijackGradientsOptimizer(tf.train.AdamOptimizer(learning_rate))

            self._train_op = optimizer.minimize(loss)
            self._grad_op = optimizer._grad_op
            self._sess = tf.Session()

    def forward(self, x, y):
        _, grad = self._sess.run([self._train_op, self._grad_op], feed_dict = {self._inputs:x, self._labels:y})    
        return grad

    def conv_layer(self, prev_layer, layer_depth):
        strides = 2 if layer_depth % 3 == 0 else 1
        conv_layer = tf.layers.conv2d(prev_layer, layer_depth * 4, 3, strides, 'same', activation=tf.nn.relu)
        return conv_layer

    def fully_connected(self, prev_layer, num_units):
        layer = tf.layers.dense(prev_layer, num_units, activation=tf.nn.relu)
        return layer

def main():
    ps = ParameterServer()
    learning_rate = 0.02
    batch_size = 64
    worker1 = Worker("worker1", ps, DataSource(batch_size), KerasCNN, learning_rate, batch_size).run()
    worker2 = Worker("worker2", ps, DataSource(batch_size), KerasCNN, learning_rate, batch_size).run()
    worker1.join()
    worker2.join()

if __name__ == '__main__':
    main()
