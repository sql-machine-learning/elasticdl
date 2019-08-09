import tensorflow as tf
from elasticdl.python.common.constants import Mode


class CustomModel(tf.keras.Model):
    def __init__(self, channel_last=True):
        super(CustomModel, self).__init__(name="mnist_model")
        if channel_last:
            self._reshape = tf.keras.layers.Reshape((28, 28, 1))
        else:
            self._reshape = tf.keras.layers.Reshape((1, 28, 28))
        self._conv1 = tf.keras.layers.Conv2D(
            32, kernel_size=(3, 3), activation="relu"
        )
        self._conv2 = tf.keras.layers.Conv2D(
            64, kernel_size=(3, 3), activation="relu"
        )
        self._batch_norm = tf.keras.layers.BatchNormalization()
        self._maxpooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self._dropout = tf.keras.layers.Dropout(0.25)
        self._flatten = tf.keras.layers.Flatten()
        self._dense = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self._reshape(inputs["image"])
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._batch_norm(x, training=training)
        x = self._maxpooling(x)
        if training:
            x = self._dropout(x, training=training)
        x = self._flatten(x)
        x = self._dense(x)
        return x


def loss(output, labels):
    labels = tf.reshape(labels, [-1])
    return tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output, labels=labels
        )
    )


def optimizer(lr=0.01):
    return tf.optimizers.SGD(lr)


def dataset_fn(dataset, mode):
    if mode == Mode.PREDICTION:
        raise Exception(
            "dataset_fn in prediction mode is not implemented for this model yet."
        )

    def _parse_data(record):
        feature_description = {
            "image": tf.io.FixedLenFeature([28, 28], tf.float32),
            "label": tf.io.FixedLenFeature([1], tf.int64),
        }
        r = tf.io.parse_single_example(record, feature_description)
        label = r["label"]
        image = r["image"]
        return image, label

    dataset = dataset.map(_parse_data)
    dataset = dataset.map(
        lambda x, y: (
            {"image": tf.math.divide(tf.cast(x, tf.float32), 255.0)},
            tf.cast(y, tf.int32),
        )
    )
    if mode != Mode.TRAINING:
        dataset = dataset.shuffle(buffer_size=1024)
    return dataset


def eval_metrics_fn(predictions, labels):
    labels = tf.reshape(labels, [-1])
    return {
        "accuracy": tf.reduce_mean(
            input_tensor=tf.cast(
                tf.equal(
                    tf.argmax(predictions, 1, output_type=tf.dtypes.int32),
                    labels,
                ),
                tf.float32,
            )
        )
    }
