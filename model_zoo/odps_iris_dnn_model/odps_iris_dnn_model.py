import tensorflow as tf


def custom_model():
    inputs = tf.keras.layers.Input(shape=(4, 1), name="input")
    outputs = tf.keras.layers.Dense(3, name="output")(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="simple-model")


def loss(output, labels):
    return tf.reduce_sum(tf.reduce_mean(tf.reshape(output, [-1])) - labels)


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def dataset_fn(dataset, _):
    def _parse_data(record):
        return tf.reshape(record[0:3], (3, 1)), tf.reshape(record[4], (1,))

    dataset = dataset.map(_parse_data)
    return dataset


def eval_metrics_fn():
    return {
        "dummy_metric": lambda labels, predictions: tf.reduce_sum(
            tf.reduce_mean(tf.reshape(predictions, [-1])) - labels
        )
    }

