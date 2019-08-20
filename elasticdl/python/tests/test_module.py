import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input

from elasticdl.python.common.constants import Mode
from elasticdl.python.model import ElasticDLKerasBaseModel


class CustomModel(ElasticDLKerasBaseModel):
    def __init__(self, context=None, **kwargs):
        super(CustomModel, self).__init__(context=context)
        self._model = self.custom_model()

    def custom_model(self):
        inputs = Input(shape=(1, 1), name="x")
        outputs = Dense(1)(inputs)
        return Model(inputs, outputs)

    def call(self, inputs, training=False):
        return self._model.call(inputs, training=training)

    def get_model(self):
        return self._model

    def loss(self, outputs, labels):
        return tf.reduce_mean(tf.square(outputs - labels))

    def optimizer(self, lr=0.1):
        return tf.optimizers.SGD(lr)

    def metrics(
        self, mode=Mode.TRAINING, outputs=None, predictions=None, labels=None
    ):
        if mode == Mode.EVALUATION:
            return {"mse": tf.reduce_mean(tf.square(predictions - labels))}


def dataset_fn(dataset, training=True):
    def _parse_data(record):
        feature_description = {
            "x": tf.io.FixedLenFeature([1], tf.float32),
            "y": tf.io.FixedLenFeature([1], tf.float32),
        }
        r = tf.io.parse_single_example(record, feature_description)
        return {"x": r["x"]}, r["y"]

    dataset = dataset.map(_parse_data)
    return dataset
