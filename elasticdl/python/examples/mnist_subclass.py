import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
import numpy as np
import PIL.Image


class MnistModel(tf.keras.Model):
    def __init__(self, channel_last=True):
        super(MnistModel, self).__init__(name="mnist_model")
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
        x = self._reshape(inputs)
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._batch_norm(x, training=training)
        x = self._maxpooling(x)
        if training:
            x = self._dropout(x, training=training)
        x = self._flatten(x)
        x = self._dense(x)
        return x


model = MnistModel()


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
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output, labels=labels.flatten()
        )
    )


def optimizer(lr=0.01):
    return tf.optimizers.SGD(lr)


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
    images = np.reshape(images, (batch_size, 28, 28))
    image_tensor = tf.convert_to_tensor(value=images)
    label_nparray = np.array(label_list)
    return ([image_tensor], label_nparray)


def eval_metrics_fn(predictions, labels):
    return {
        "accuracy": tf.reduce_mean(
            input_tensor=tf.cast(
                tf.equal(tf.argmax(predictions, 1), labels.flatten()),
                tf.float32,
            )
        )
    }
