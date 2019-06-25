import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
import numpy as np


inputs = tf.keras.layers.Input(shape=(32, 32, 3), name="image")
use_bias = True

conv = tf.keras.layers.Conv2D(
    32, kernel_size=(3, 3), padding="same", use_bias=use_bias, activation=None
)(inputs)
bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(
    conv
)
activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

conv = tf.keras.layers.Conv2D(
    32, kernel_size=(3, 3), padding="same", use_bias=use_bias, activation=None
)(activation)
bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(
    conv
)
activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(activation)
dropout = tf.keras.layers.Dropout(0.2)(max_pool)

conv = tf.keras.layers.Conv2D(
    64, kernel_size=(3, 3), padding="same", use_bias=use_bias, activation=None
)(dropout)
bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(
    conv
)
activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

conv = tf.keras.layers.Conv2D(
    64, kernel_size=(3, 3), padding="same", use_bias=use_bias, activation=None
)(activation)
bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(
    conv
)
activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

max_pool = tf.keras.layers.MaxPooling2D()(activation)
dropout = tf.keras.layers.Dropout(0.3)(max_pool)

conv = tf.keras.layers.Conv2D(
    128, kernel_size=(3, 3), padding="same", use_bias=use_bias, activation=None
)(dropout)
bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(
    conv
)
activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

conv = tf.keras.layers.Conv2D(
    128, kernel_size=(3, 3), padding="same", use_bias=use_bias, activation=None
)(activation)
bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(
    conv
)
activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

max_pool = tf.keras.layers.MaxPooling2D()(activation)
dropout = tf.keras.layers.Dropout(0.4)(max_pool)

flatten = tf.keras.layers.Flatten()(dropout)
outputs = tf.keras.layers.Dense(10, name="output")(flatten)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cifar10_model")


def data_schema():
    """
    list of dicts which include name, shape, dtype.
    """
    return [
        {"name": "image", "shape": [32, 32, 3], "dtype": tf.dtypes.float32},
        {"name": "label", "shape": [1], "dtype": tf.dtypes.int64},
    ]


def loss(output, labels):
    return tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output, labels=labels.flatten()
        )
    )


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def input_fn(records):
    image_list = []
    label_list = []
    # deserialize
    for r in records:
        get_np_val = (
            lambda data: data.numpy()
            if isinstance(data, EagerTensor)
            else data
        )
        label = get_np_val(r["label"])
        image = get_np_val(r["image"])
        image = image.astype(np.float32)
        image /= 255
        label = label.astype(np.int32)
        image_list.append(image)
        label_list.append(label)

    # batching
    batch_size = len(image_list)
    images = np.concatenate(image_list, axis=0)
    images = np.reshape(images, (batch_size, 32, 32, 3))
    images = tf.convert_to_tensor(value=images)
    labels = np.array(label_list)
    return ({"image": images}, labels)


def eval_metrics_fn(predictions, labels):
    return {
        "accuracy": tf.reduce_mean(
            input_tensor=tf.cast(
                tf.equal(
                    tf.argmax(input=predictions, axis=1), labels.flatten()
                ),
                tf.float32,
            )
        )
    }
