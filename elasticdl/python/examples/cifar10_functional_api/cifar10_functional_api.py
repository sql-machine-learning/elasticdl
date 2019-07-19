import numpy as np
import tensorflow as tf


def custom_model():
    inputs = tf.keras.layers.Input(shape=(32, 32, 3), name="image")
    use_bias = True

    conv = tf.keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
        use_bias=use_bias,
        activation=None,
    )(inputs)
    bn = tf.keras.layers.BatchNormalization(
        epsilon=1e-06, axis=-1, momentum=0.9
    )(conv)
    activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

    conv = tf.keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        padding="same",
        use_bias=use_bias,
        activation=None,
    )(activation)
    bn = tf.keras.layers.BatchNormalization(
        epsilon=1e-06, axis=-1, momentum=0.9
    )(conv)
    activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

    max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(activation)
    dropout = tf.keras.layers.Dropout(0.2)(max_pool)

    conv = tf.keras.layers.Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        use_bias=use_bias,
        activation=None,
    )(dropout)
    bn = tf.keras.layers.BatchNormalization(
        epsilon=1e-06, axis=-1, momentum=0.9
    )(conv)
    activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

    conv = tf.keras.layers.Conv2D(
        64,
        kernel_size=(3, 3),
        padding="same",
        use_bias=use_bias,
        activation=None,
    )(activation)
    bn = tf.keras.layers.BatchNormalization(
        epsilon=1e-06, axis=-1, momentum=0.9
    )(conv)
    activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

    max_pool = tf.keras.layers.MaxPooling2D()(activation)
    dropout = tf.keras.layers.Dropout(0.3)(max_pool)

    conv = tf.keras.layers.Conv2D(
        128,
        kernel_size=(3, 3),
        padding="same",
        use_bias=use_bias,
        activation=None,
    )(dropout)
    bn = tf.keras.layers.BatchNormalization(
        epsilon=1e-06, axis=-1, momentum=0.9
    )(conv)
    activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

    conv = tf.keras.layers.Conv2D(
        128,
        kernel_size=(3, 3),
        padding="same",
        use_bias=use_bias,
        activation=None,
    )(activation)
    bn = tf.keras.layers.BatchNormalization(
        epsilon=1e-06, axis=-1, momentum=0.9
    )(conv)
    activation = tf.keras.layers.Activation(tf.nn.relu)(bn)

    max_pool = tf.keras.layers.MaxPooling2D()(activation)
    dropout = tf.keras.layers.Dropout(0.4)(max_pool)

    flatten = tf.keras.layers.Flatten()(dropout)
    outputs = tf.keras.layers.Dense(10, name="output")(flatten)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="cifar10_model")


def loss(output, labels):
    labels = tf.reshape(labels, [-1])
    return tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output, labels=labels
        )
    )


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def dataset_fn(dataset):
    def _parse_data(record):
        feature_description = {
            "image": tf.io.FixedLenFeature([32, 32, 3], tf.float32),
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
            y,
        )
    )
    return dataset


def input_fn(records):
    feature_description = {
        "image": tf.io.FixedLenFeature([32, 32, 3], tf.float32),
        "label": tf.io.FixedLenFeature([1], tf.int64),
    }
    image_list = []
    label_list = []
    for r in records:
        # deserialization
        r = tf.io.parse_single_example(r, feature_description)
        label = r["label"].numpy()
        image = r["image"].numpy()
        # processing data
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
    labels = tf.reshape(labels, [-1])
    return {
        "accuracy": tf.reduce_mean(
            input_tensor=tf.cast(
                tf.equal(tf.argmax(input=predictions, axis=1), labels),
                tf.float32,
            )
        )
    }
