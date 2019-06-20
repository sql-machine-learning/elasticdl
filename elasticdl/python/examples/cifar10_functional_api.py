import tensorflow as tf
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


def loss(output, labels):
    return tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output, labels=labels.flatten()
        )
    )


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


feature_label_colums = [
    tf.feature_column.numeric_column(
        key="image", dtype=tf.dtypes.float32, shape=[32, 32, 3]
    ),
    tf.feature_column.numeric_column(
        key="label", dtype=tf.dtypes.int64, shape=[1]
    )
]
feature_spec = tf.feature_column.make_parse_example_spec(feature_label_colums)


def input_fn(record_list, decode_fn):
    image_numpy_list = []
    label_list = []
    # deserialize
    for r in record_list:
        tensor_dict = decode_fn(r, feature_spec)
        label = tensor_dict['label'].numpy().astype(np.int32)
        label_list.append(label)

        image_numpy = tensor_dict['image'].numpy().astype(np.float32) / 255
        image_numpy_list.append(image_numpy)

    # batching
    batch_size = len(image_numpy_list)
    images = np.concatenate(image_numpy_list, axis=0)
    images = np.reshape(images, (batch_size, 32, 32, 3))
    image_tensor = tf.convert_to_tensor(value=images)
    label_nparray = np.array(label_list)
    return ([image_tensor], label_nparray)


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
