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

import horovod.tensorflow as hvd
import tensorflow as tf

from elasticdl.python.common.constants import Mode
from elasticdl.python.common.log_utils import default_logger as logger


def train(dataset, elastic_controller):
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
    optimizer = hvd.DistributedOptimizer(optimizer)
    train_step = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Use the elastic wrapper to wrap the function to train one batch
        elastic_train_one_batch = elastic_controller.elastic_run(
            train_one_batch
        )
        for i in range(1000):
            loss_value, _ = elastic_train_one_batch(sess, [loss, train_step])
            logger.info("loss: {}".format(loss_value))


def train_one_batch(sess, run_tensors):
    return sess.run(run_tensors)


def feed(dataset, mode, _):
    dataset = dataset.map(_parse_data)

    if mode == Mode.TRAINING:
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
