import argparse
import tensorflow as tf

from elasticdl.python.common.save_utils import CheckpointSaver
from elasticdl.python.ps.parameter_server import Parameters
from elasticdl.python.tests.test_utils import (
    save_checkpoint_without_embedding
)


def mnist_custom_model():
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


def add_params(parser):
    parser.add_argument(
        "--checkpoint_dir",
        help="The directory to store the mnist checkpoint",
        default="",
        type=str,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_params(parser)
    args, _ = parser.parse_known_args()
    print(args)
    model = mnist_custom_model()
    save_checkpoint_without_embedding(model, args.checkpoint_dir)
