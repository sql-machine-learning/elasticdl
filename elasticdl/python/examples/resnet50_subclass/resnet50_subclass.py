import numpy as np
import tensorflow as tf
from resnet50.model import ResNet50 


model = ResNet50(num_classes=10, dtype='float32') 


def loss(output, labels):
    return tf.reduce_mean(
        input_tensor=tf.keras.losses.sparse_categorical_crossentropy(
            logits=output, labels=labels.flatten()
        )
    )


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def input_fn(records):
    feature_description = {
        "image": tf.io.FixedLenFeature([224, 224, 3], tf.float32),
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
    images = np.reshape(images, (batch_size, 224, 224, 3))
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
