import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Flatten

from elasticdl.python.elasticdl.layers.embedding import Embedding


class CustomModel(tf.keras.Model):
    def __init__(self, embedding_dim=16):
        super(CustomModel, self).__init__(name="embedding_test_model")
        self.embedding_dim = embedding_dim
        self.embedding_1 = Embedding(embedding_dim)
        self.embedding_2 = Embedding(embedding_dim)
        self.concat = Concatenate()
        self.dense = Dense(1, input_shape=(embedding_dim * 3,))
        self.flatten = Flatten()

    def call(self, inputs, training=False):
        f1 = self.embedding_1(inputs["f1"])
        f2 = self.embedding_1(inputs["f2"])
        f3 = self.embedding_2(inputs["f3"])
        x = self.concat([f1, f2, f3])
        x = self.dense(x)
        return self.flatten(x)


def loss(predictions, labels):
    return tf.reduce_mean(tf.square(predictions - labels))


def dataset_fn(dataset, training=True):
    def _parse_data(record):
        feature_description = {
            "f1": tf.io.FixedLenFeature([1], tf.int64),
            "f2": tf.io.FixedLenFeature([1], tf.int64),
            "f3": tf.io.FixedLenFeature([1], tf.int64),
            "label": tf.io.FixedLenFeature([1], tf.int64),
        }
        r = tf.io.parse_single_example(record, feature_description)
        return {"f1": r["f1"], "f2": r["f2"], "f3": r["f3"]}, r["label"]

    dataset = dataset.map(_parse_data)
    return dataset


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def eval_metrics_fn(predictions, labels):
    return {"mse": tf.reduce_mean(tf.square(predictions - labels))}
