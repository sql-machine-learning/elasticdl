import tensorflow as tf
import numpy as np 
import itertools

class Elastic(tf.train.Optimizer):
    def __init__(self, optimizer):
        self._optimizer = optimizer
        # TODO(l.zou): Need to understand use_locking
        super(Elastic, self).__init__(name = "Elastic", use_locking = False)

    # The method we want to intercept
    def compute_gradients(self, *args, **kwargs):
        gradients = self._optimizer.compute_gradients(*args, **kwargs)
        # Modify graph to add print OP to all vars and gradients .
        # TODO(l.zou): need to write OP to send gradients to PS.
        modified_grad = []
        for grad, var in gradients:
            grad = tf.Print(grad, [grad])
            modified_grad.append((grad, var))
        return modified_grad

    # Forward all other methods. TODO(l.zou): could use a proxy to automate these
    def get_slot(self, *args, **kwargs):
        return self._optimizer.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        return self._optimizer.get_slot_names(*args, **kwargs)

    def variables(self, *args, **kwargs):
        return self._optimizer.variables(*args, **kwargs)

    def apply_gradients(self, *args, **kwargs):
        return self._optimizer.apply_gradients(*args, **kwargs)

# training data
x_data = np.random.rand(1000)
y_data = x_data * 2 + 1

# simple linear model
x = tf.placeholder("float")
y = tf.placeholder("float")
w = tf.Variable([0.0, 0.0], name = "w")
y_predict = tf.multiply(x, w[0]) + w[1]

loss = tf.square(y - y_predict)

optimizer = Elastic(tf.train.GradientDescentOptimizer(0.1))
train_op = optimizer.minimize(loss)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for sx, sy in itertools.chain(*[zip(x_data, y_data)]*10):
        sess.run(train_op, feed_dict={x:sx, y:sy}) 
        sw = sess.run(w)
        print('----', sw)