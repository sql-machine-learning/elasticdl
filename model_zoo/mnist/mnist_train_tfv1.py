# Copyright 2020 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import tensorflow as tf

from elasticai_api.io.recordio_reader import RecordIOReader
from elasticai_api.tensorflow.controller import create_elastic_controller
from elasticai_api.tensorflow.optimizer import (
    AdjustBackwardPassesPerStepHook,
    DistributedOptimizer,
)
from elasticdl.python.common.log_utils import default_logger as logger

layers = tf.layers


def get_dataset_gen(data_shard_service, data_reader):
    def gen():
        while True:
            shard = data_shard_service.fetch_shard()
            if not shard:
                raise StopIteration("No data")
            count = 0
            for record in data_reader.read_records(shard.start, shard.end):
                count += 1
                yield record

    return gen


def create_dataset(data_shard_service, training_data_dir):
    data_files = _get_data_files(training_data_dir)
    data_reader = RecordIOReader(data_files)
    gen = get_dataset_gen(data_shard_service, data_reader)
    dataset = tf.data.Dataset.from_generator(gen, tf.string)
    return dataset


def _get_data_files(data_dir):
    data_files = []
    for filename in os.listdir(data_dir):
        data_files.append(os.path.join(data_dir, filename))
    return data_files


def conv_model(feature, target, mode):
    """2-layer convolution model."""
    # Convert the target to a one-hot tensor of shape (batch_size, 10) and
    # with a on-value of 1 for each one-hot vector of length 10.
    target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)

    # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
    # image width and height final dimension being the number of color
    # channels.
    feature = tf.reshape(feature, [-1, 28, 28, 1])

    # First conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope("conv_layer1"):
        h_conv1 = layers.conv2d(
            feature,
            32,
            kernel_size=[5, 5],
            activation=tf.nn.relu,
            padding="SAME",
        )
        h_pool1 = tf.nn.max_pool(
            h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )

    # Second conv layer will compute 64 features for each 5x5 patch.
    with tf.variable_scope("conv_layer2"):
        h_conv2 = layers.conv2d(
            h_pool1,
            64,
            kernel_size=[5, 5],
            activation=tf.nn.relu,
            padding="SAME",
        )
        h_pool2 = tf.nn.max_pool(
            h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # Densely connected layer with 1024 neurons.
    h_fc1 = layers.dropout(
        layers.dense(h_pool2_flat, 1024, activation=tf.nn.relu),
        rate=0.5,
        training=mode == tf.estimator.ModeKeys.TRAIN,
    )

    # Compute logits (1 per class) and compute loss.
    logits = layers.dense(h_fc1, 10, activation=None)
    loss = tf.losses.softmax_cross_entropy(target, logits)

    return tf.argmax(logits, 1), loss


def train(args):
    allreduce_controller = create_elastic_controller(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        dataset_size=50000,
    )
    dataset = create_dataset(
        allreduce_controller.data_shard_service, args.training_data
    )
    dataset = feed(dataset)
    dataset = dataset.batch(args.batch_size).prefetch(1)
    dataset_it = dataset.make_one_shot_iterator()
    batch_x, batch_y = dataset_it.get_next()
    batch_x = tf.cast(batch_x, tf.float32)

    batch_y = tf.reshape(batch_y, (-1,))
    image = tf.reshape(batch_x, (-1, 784))
    predict, loss = conv_model(image, batch_y, tf.estimator.ModeKeys.TRAIN)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = DistributedOptimizer(optimizer, fixed_global_batch_size=True)
    global_step = tf.train.get_or_create_global_step()
    train_step = optimizer.minimize(loss, global_step=global_step)

    # Use the elastic wrapper to wrap the function to train one batch
    elastic_train_one_batch = allreduce_controller.elastic_run(train_one_batch)
    hook = AdjustBackwardPassesPerStepHook(optimizer)
    allreduce_controller.set_broadcast_variables(tf.global_variables())
    with allreduce_controller.scope():
        with tf.train.MonitoredTrainingSession(hooks=[hook]) as sess:
            allreduce_controller.set_session(sess)
            try:
                while True:
                    loss_value, step, _ = elastic_train_one_batch(
                        sess, [loss, global_step, train_step]
                    )
                    logger.info(
                        "global step = {}. loss: {}".format(step, loss_value)
                    )
            except tf.errors.OutOfRangeError:
                print("end!")


def train_one_batch(sess, run_tensors):
    return sess.run(run_tensors)


def feed(dataset):
    dataset = dataset.map(_parse_data)
    dataset = dataset.shuffle(buffer_size=1024)
    return dataset


def _parse_data(record):
    feature_description = {
        "image": tf.io.FixedLenFeature([28, 28], tf.float32),
        "label": tf.io.FixedLenFeature([1], tf.int64),
    }
    r = tf.io.parse_single_example(record, feature_description)
    features = {
        "image": tf.math.divide(tf.cast(r["image"], tf.float32), 255.0)
    }
    return features["image"], tf.cast(r["label"], tf.int32)


def eval_metrics_fn():
    return {
        "accuracy": lambda labels, predictions: tf.equal(
            tf.argmax(predictions, 1, output_type=tf.int32),
            tf.cast(tf.reshape(labels, [-1]), tf.int32),
        )
    }


def arg_parser():
    parser = argparse.ArgumentParser(description="Process training parameters")
    parser.add_argument("--batch_size", type=int, default=64, required=False)
    parser.add_argument("--num_epochs", type=int, default=1, required=False)
    parser.add_argument(
        "--learning_rate", type=float, default=0.1, required=False
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disable CUDA training",
    )
    parser.add_argument("--training_data", type=str, required=True)
    parser.add_argument(
        "--validation_data", type=str, default="", required=False
    )
    return parser


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    print(args)
    train(args)
