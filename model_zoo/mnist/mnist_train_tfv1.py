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
import recordio
from contextlib import closing
import horovod.tensorflow as hvd
import tensorflow as tf

from elasticdl.python.common.log_utils import default_logger as logger
from elasticai_api.tensorflow.controller import create_elastic_controller
from elasticai_api.tensorflow.optimizer import DistributedOptimizer


def get_dataset_gen(data_shard_service):
    def gen():
        while(True):
            shard = data_shard_service.fetch_shard()
            if not shard:
                raise StopIteration("No data")
            with closing(
                recordio.Scanner(
                    shard.name,
                    shard.start,
                    shard.end - shard.start,
                )
            ) as reader:
                for i in range(shard.start, shard.end):
                    record = reader.record()
                    if record:
                        yield record
    return gen


def create_dataset(data_shard_service):
    gen = get_dataset_gen(data_shard_service)
    dataset = tf.data.Dataset.from_generator(gen, tf.string)
    return dataset


def train(args):
    allreduce_controller = create_elastic_controller(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        training_data=args.training_data,
    )
    dataset = create_dataset(allreduce_controller.data_shard_service)
    dataset = feed(dataset)
    dataset = dataset.batch(args.batch_size).prefetch(1)
    dataset_it = dataset.make_one_shot_iterator()
    batch_x, batch_y = dataset_it.get_next()
    batch_x = tf.cast(batch_x, tf.float32)

    x = tf.keras.layers.Reshape((28, 28, 1))(batch_x)
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10)(x)
    loss = tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=outputs, labels=tf.reshape(batch_y, [-1])
        )
    )
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = DistributedOptimizer(optimizer)
    train_step = optimizer.minimize(loss)

    with allreduce_controller.scope():
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Use the elastic wrapper to wrap the function to train one batch
            elastic_train_one_batch = allreduce_controller.elastic_run(
                train_one_batch
            )
            for i in range(1000):
                loss_value, _ = elastic_train_one_batch(sess, [loss, train_step])
                logger.info("loss: {}".format(loss_value))


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
