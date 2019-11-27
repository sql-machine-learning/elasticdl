import tensorflow as tf


def custom_model():
    inputs = tf.keras.layers.Input(shape=(4, 1), name="input")
    x = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(3, name="output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="simple-model")


def loss(labels, predictions):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            tf.cast(tf.reshape(labels, [-1]), tf.int32), predictions
        )
    )


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def eval_metrics_fn():
    return {
        "accuracy": lambda labels, predictions: tf.equal(
            tf.argmax(predictions, 1, output_type=tf.int32),
            tf.cast(tf.reshape(labels, [-1]), tf.int32),
        )
    }
