import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    Flatten,
    Layer,
    Multiply,
    Subtract,
)

from elasticdl.python.common.constants import Mode

AUC_metric = None


class ApplyMask(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(ApplyMask, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(ApplyMask, self).build(
            input_shape
        )  # Be sure to call this somewhere!

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return Multiply()([x, tf.cast(K.expand_dims(mask, -1), tf.float32)])

    def compute_output_shape(self, input_shape):
        return input_shape


def custom_model(
    input_dim=5383, embedding_dim=64, input_length=10, fc_unit=64
):
    inputs = tf.keras.Input(shape=(input_length,))
    embed_layer = Embedding(
        input_dim=input_dim,
        output_dim=embedding_dim,
        mask_zero=True,
        input_length=input_length,
    )
    embeddings = embed_layer(inputs)
    embeddings = ApplyMask()(embeddings)

    emb_sum = K.sum(embeddings, axis=1)
    emb_sum_square = K.square(emb_sum)
    emb_square = K.square(embeddings)
    emb_square_sum = K.sum(emb_square, axis=1)
    second_order = K.sum(
        0.5 * Subtract()([emb_sum_square, emb_square_sum]), axis=1
    )

    id_bias = Embedding(input_dim=input_dim, output_dim=1, mask_zero=True)(
        inputs
    )
    id_bias = ApplyMask()(id_bias)
    first_order = K.sum(id_bias, axis=(1, 2))
    fm_output = tf.keras.layers.Add()([first_order, second_order])

    nn_input = Flatten()(embeddings)
    nn_h = Dense(fc_unit)(nn_input)
    deep_output = Dense(1)(nn_h)
    deep_output = tf.reshape(deep_output, shape=(-1,))
    outputs = tf.keras.layers.Add()([fm_output, deep_output])

    m = tf.keras.Model(inputs=inputs, outputs=outputs)

    global AUC_metric
    AUC_metric = tf.keras.metrics.AUC()

    return m


def loss(output, labels):
    labels = tf.cast(tf.reshape(labels, [-1]), tf.dtypes.float32)
    output = tf.reshape(output, [-1])
    return tf.reduce_mean(
        input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
            logits=output, labels=labels
        )
    )


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def dataset_fn(dataset, mode):
    def _parse_data(record):
        if mode == Mode.PREDICTION:
            feature_description = {
                "feature": tf.io.FixedLenFeature([10], tf.int64)
            }
        else:
            feature_description = {
                "feature": tf.io.FixedLenFeature([10], tf.int64),
                "label": tf.io.FixedLenFeature([1], tf.int64),
            }
        r = tf.io.parse_single_example(record, feature_description)
        features = {"feature": tf.cast(r["feature"], tf.float32)}
        if mode == Mode.PREDICTION:
            return features
        return features, tf.cast(r["label"], tf.int32)

    dataset = dataset.map(_parse_data)

    if mode != Mode.PREDICTION:
        dataset = dataset.shuffle(buffer_size=1024)
    return dataset


def eval_metrics_fn(predictions, labels):
    labels = tf.reshape(labels, [-1])
    predictions = tf.reshape(predictions, [-1])
    global AUC_metric
    return {
        "accuracy": tf.reduce_mean(
            input_tensor=tf.cast(
                tf.equal(tf.cast(predictions > 0.0, tf.dtypes.int32), labels),
                tf.float32,
            )
        ),
        "auc": AUC_metric(labels, tf.sigmoid(predictions)),
    }
