import tensorflow as tf
import numpy as np

class MnistCNN(object):

    def __init__(self):
        super(MnistCNN, self).__init__()

    @staticmethod
    def forward(input):
        # Simple CNN for MNIST
        conv1 = tf.layers.conv2d(
            input, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        fc = tf.contrib.layers.flatten(conv2)
        fc = tf.layers.dense(fc, 1024)
        fc = tf.layers.dropout(fc, rate=0.75, training=True)
        logits = tf.layers.dense(fc, 10)
        return logits

    @staticmethod
    def loss(x, y):
        one_hot = tf.one_hot(y, 10)
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=x,
                labels=one_hot))

    @staticmethod
    def optimizer():
        return tf.train.AdamOptimizer(learning_rate=0.001)

    @staticmethod
    def accuracy(output, labels):
        correct_prediction = tf.equal(
            tf.argmax(
                    output, 1), tf.argmax(
                    tf.one_hot(labels, 10), 1))
        return tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))

    def input_tensors(self):
        return tf.placeholder(tf.float32, [None, 28, 28, 1])

    @staticmethod
    def raw_data_transform_by_py(databytes):
        N = 28
        parsed = np.frombuffer(databytes, dtype="uint8")
        assert len(parsed) == N * N + 1
        label = parsed[-1]
        img = np.resize(parsed[:-1], new_shape=(N, N, 1))
        label = label.astype(np.int32)
        img = img.astype(np.float32) / 255.0
        
        return img, label

    @staticmethod
    def transformed_data_types():
        return [tf.float32, tf.int32]

    @staticmethod
    def data_preprocess_by_tf(image, label):
        N = 28
        image = tf.reshape(image, [N, N, 1])
        return image, label
