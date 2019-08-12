import numpy as np
import PIL.Image
import tensorflow as tf

from elasticdl.python.common.constants import Mode


def custom_model():
    inputs = tf.keras.Input(shape=(28, 28), name="image")
    x = tf.keras.layers.Reshape((28, 28, 1))(inputs)
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")


def prepare_data_for_a_single_file(file_object, filename):
    """
    :param filename: training data file name
    :param file_object: a file object associated with filename
    """
    label = int(filename.split("/")[-2])
    image = PIL.Image.open(file_object)
    numpy_image = np.array(image)
    example_dict = {
        "image": tf.train.Feature(
            float_list=tf.train.FloatList(value=numpy_image.flatten())
        ),
        "label": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[label])
        ),
    }
    example = tf.train.Example(
        features=tf.train.Features(feature=example_dict)
    )
    return example.SerializeToString()


def loss(output, labels):
    labels = tf.reshape(labels, [-1])
    return tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output, labels=labels
        )
    )


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def dataset_fn(dataset, mode):
    def _parse_data(record):
        if mode == Mode.PREDICTION:
            feature_description = {
                "image": tf.io.FixedLenFeature([28, 28], tf.float32)
            }
        else:
            feature_description = {
                "image": tf.io.FixedLenFeature([28, 28], tf.float32),
                "label": tf.io.FixedLenFeature([1], tf.int64),
            }
        r = tf.io.parse_single_example(record, feature_description)
        features = {
            "image": tf.math.divide(tf.cast(r["image"], tf.float32), 255.0)
        }
        if mode == Mode.PREDICTION:
            return features
        else:
            return features, tf.cast(r["label"], tf.int32)

    dataset = dataset.map(_parse_data)

    if mode != Mode.PREDICTION:
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
