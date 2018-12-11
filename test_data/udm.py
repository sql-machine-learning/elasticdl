import tensorflow as tf

class UserDefinedModule(object):

    def __init__(self):
        self._vars = []  

    def forward(self, x):
        pass

    def loss(self, x, y):
        raise NotImplementedError()

    def optimizer(self, *args, **kwargs):
        raise NotImplementedError()
 
    def trainable_variables(self):
        raise NotImplementedError()

    def vars(self):
        self._init_vars()
        return self._vars
        
    def _init_vars(self):
        graph = tf.Graph() 
        with graph.as_default():
            input = tf.placeholder(tf.float32, [None, 28, 28, 1])
            self.forward(input)
            trainable_vars = tf.trainable_variables()
            init_op = tf.initializers.global_variables()
        var_names = [v.name.split(":", 1)[0] for v in trainable_vars]

        with graph.as_default(), tf.Session() as sess:
            sess.run(init_op)
            var_values = sess.run(trainable_vars)
        self._vars = dict(zip(var_names, var_values))
