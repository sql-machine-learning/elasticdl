import tensorflow as tf
import numpy as np
import PIL.Image


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


def prepare_data_for_a_single_file(file_object, filename):
    """
    :param filename: training data file name
    :param file_object: a file object associated with filename
    :return: an exmaple object
    """
    label = int(filename.split("/")[-2])
    image = PIL.Image.open(file_object)
    numpy_image = np.array(image)
    feature_name_to_feature = {}
    feature_name_to_feature['image'] = tf.train.Feature(
        float_list=tf.train.FloatList(
            value=numpy_image.astype(tf.float32.as_numpy_dtype).flatten(),
        ),
    )
    feature_name_to_feature['label'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[label]),
    )
    return tf.train.Example(
        features=tf.train.Features(feature=feature_name_to_feature),
    )


def loss(output, labels):
    return tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output, labels=labels.flatten()
        )
    )


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def input_fn(record_list, decode_fn):
    image_numpy_list = []
    label_list = []
    # deserialize
    for r in record_list:
        example = decode_fn(r)

        image_array = example.features.feature['image'].float_list.value
        image_numpy = np.asarray(image_array).reshape(28, 28)
        image_numpy_list.append(image_numpy.astype(np.float32) / 255)

        label = example.features.feature['label'].int64_list.value[0]
        label_list.append(label)

    # batching
    batch_size = len(image_numpy_list)
    images = np.concatenate(image_numpy_list, axis=0)
    images = np.reshape(images, (batch_size, 28, 28))
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
