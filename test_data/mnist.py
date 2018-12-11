import tensorflow as tf
from udm import UserDefinedModule

class MnistCNN(UserDefinedModule):

    def __init__(self):
        super(MnistCNN, self).__init__()

    def forward(self, input):
        # Simple CNN for MNIST
        conv1 = tf.layers.conv2d(
            input, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        fc = tf.contrib.layers.flatten(conv2)
        fc = tf.layers.dense(fc, 1024)
        fc = tf.layers.dropout(fc, rate=0.75, training=True)
        self._logits = tf.layers.dense(fc, 10)
        return self._logits

    def loss(self, x, y):
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x,
                labels=y))

    def optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer(learning_rate=0.01)
