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

"""
Download the mnist dataset from
https://s3.amazonaws.com/fast-ai-imageclas/mnist_png.tgz
and then untar it into ${data_store_dir}. On minikube, we can use the
following command to submit a training job with these codes.

elasticdl train \
  --image_name=elasticdl:pt_mnist_allreduce  \
  --job_command="python -m model_zoo.mnist.mnist_tfv1_png \
      --training_data=/local_data/mnist_png/training/"
  --num_minibatches_per_task=2 \
  --num_workers=2 \
  --worker_pod_priority=0.5 \
  --master_resource_request="cpu=0.2,memory=1024Mi" \
  --master_resource_limit="cpu=1,memory=2048Mi" \
  --worker_resource_request="cpu=0.3,memory=1024Mi" \
  --worker_resource_limit="cpu=1,memory=2048Mi" \
  --envs="PYTHONUNBUFFERED=0,HOROVOD_ELASTIC=1" \
  --job_name=test-mnist-allreduce \
  --image_pull_policy=Never \
  --volume="host_path=${data_store_dir},mount_path=/local_data" \
  --distribution_strategy=AllreduceStrategy \
"""

import argparse
import os

import cv2
import numpy as np
import tensorflow as tf

from elasticai_api.tensorflow.controller import create_elastic_controller
from elasticai_api.tensorflow.optimizer import (
    AdjustBackwardPassesPerStepHook,
    DistributedOptimizer,
)
from elasticdl.python.common.log_utils import default_logger as logger

layers = tf.layers


def get_samples_from_folder(folder_dir):
    category_index = 0
    samples = []
    for category_name in sorted(os.listdir(folder_dir)):
        category_dir = os.path.join(folder_dir, category_name)
        if not os.path.isdir(category_dir):
            continue
        for img_file in os.listdir(category_dir):
            img_dir = os.path.join(category_dir, img_file)
            if os.path.isfile(img_dir) and img_dir.endswith("png"):
                samples.append((img_dir, category_index))
        category_index += 1
    return samples


def get_dataset_gen(data_shard_service, samples):
    def gen():
        while True:
            index = data_shard_service.fetch_record_index()
            if not index:
                raise StopIteration("No data")
            image_path, label = samples[index]
            image = cv2.imread(image_path)
            image = np.array(image / 255.0, np.float32)
            yield image, np.array([label])

    return gen


def create_dataset(data_shard_service, samples):
    gen = get_dataset_gen(data_shard_service, samples)
    dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))
    return dataset


def conv_model(feature, target, mode):
    """2-layer convolution model."""
    # Convert the target to a one-hot tensor of shape (batch_size, 10) and
    # with a on-value of 1 for each one-hot vector of length 10.
    target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)

    # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
    # image width and height final dimension being the number of color
    # channels.
    feature = tf.reshape(feature, [-1, 28, 28, 3])

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
    training_samples = get_samples_from_folder(args.training_data)
    allreduce_controller = create_elastic_controller(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        dataset_size=len(training_samples),
        shuffle=True,
    )
    dataset = create_dataset(
        allreduce_controller.data_shard_service, training_samples
    )
    dataset = dataset.batch(args.batch_size).prefetch(1)
    dataset_it = dataset.make_one_shot_iterator()
    batch_x, batch_y = dataset_it.get_next()
    batch_x = tf.cast(batch_x, tf.float32)

    batch_y = tf.reshape(batch_y, (-1,))
    predict, loss = conv_model(batch_x, batch_y, tf.estimator.ModeKeys.TRAIN)
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
