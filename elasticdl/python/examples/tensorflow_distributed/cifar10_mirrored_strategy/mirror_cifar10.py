import argparse

import tensorflow as tf
from cifar10_dataset import get_cifar10_test_dataset, get_cifar10_train_dataset


def get_model():
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
    outputs = tf.keras.layers.Softmax()(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="cifar10_model")


def optimizer(lr=0.05):
    return tf.keras.optimizers.SGD(lr)


def train_mirrored(worker_batch_size, epoch, data_path=None):

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        global_batch = (
            worker_batch_size * mirrored_strategy.num_replicas_in_sync
        )
        train_dataset = get_cifar10_train_dataset(
            global_batch, epoch, data_path
        )
        test_dataset = get_cifar10_test_dataset(global_batch, epoch, data_path)
        print("num of replica: %d" % mirrored_strategy.num_replicas_in_sync)
        print(
            "worker batch size = %d and global batch size = %d"
            % (worker_batch_size, global_batch)
        )
        model = get_model()
        opt = optimizer()
        model.compile(
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            optimizer=opt,
            metrics=["accuracy"],
        )
        model.fit(
            train_dataset,
            epochs=epoch,
            steps_per_epoch=int(50000 / global_batch),
            validation_data=test_dataset,
            validation_steps=int(10000 / global_batch),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="keras mirrored strategy")
    parser.add_argument(
        "--worker_batch_size",
        default=128,
        type=int,
        help="The batch size of of a worker",
    )
    parser.add_argument("--epoch", default=1, type=int, help="The epoch size")
    parser.add_argument("--data_path", help="cifar10 data path", default="")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_mirrored(args.worker_batch_size, args.epoch, args.data_path)


if __name__ == "__main__":
    main()
