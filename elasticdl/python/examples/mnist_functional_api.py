import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
import numpy as np


inputs = tf.keras.Input(shape=(28, 28), name="img")
x = tf.keras.layers.Reshape((28, 28, 1))(inputs)
x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(x)
x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x, training=True)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Dropout(0.25)(x, training=True)
x = tf.keras.layers.Flatten()(x)
outputs = tf.keras.layers.Dense(10)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")


def feature_columns():
    return [
        tf.feature_column.numeric_column(
            key="image", dtype=tf.dtypes.float32, shape=[28, 28]
        )
    ]


def label_columns():
    return [
        tf.feature_column.numeric_column(
            key="label", dtype=tf.dtypes.int64, shape=[1]
        )
    ]


def loss(output, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output, labels=labels.flatten()
        )
    )


def optimizer(lr=0.01):
    return tf.train.GradientDescentOptimizer(lr)


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
    images = np.reshape(images, (batch_size, 28, 28))
    labels = np.array(label_list)
    return ({"image": images}, labels)

def eval_metrics_fn(predictions, labels):
    return {
        'loss': tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=predictions, labels=labels.flatten())),
        'accuracy': tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(predictions, 1),
            labels.flatten()), tf.float32)),
    }
