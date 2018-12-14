import threading
import tensorflow as tf

class UserDefinedModule(object):

    def __init__(self):
        self._lock = threading.Lock()

    def vars(self):
        with self._lock:
            if not hasattr(self, '_vars'):
                self._init_vars()
        return self._vars
 
    def input_tensors(self):
        raise NotImplementedError()
        
    def _init_vars(self):
        graph = tf.Graph() 

        with graph.as_default():
            inputs = self.input_tensors()
            self.forward(inputs)
            trainable_vars = tf.trainable_variables()
            init_op = tf.initializers.global_variables()

        var_names = [v.name.split(":", 1)[0] for v in trainable_vars]

        with graph.as_default(), tf.Session() as sess:
           sess.run(init_op)
           var_values = sess.run(trainable_vars)
        self._vars = dict(zip(var_names, var_values))
