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
    inputs = tf.keras.Input(shape=(28, 28), name="image")
    x = tf.keras.layers.Reshape((28, 28, 1))(inputs)
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
    optimizer = tf.optimizers.SGD(0.01)

    elastic_controller.set_broadcast_model(model)
    elastic_controller.set_broadcast_optimizer(optimizer)

    for features, labels in dataset:
        elastic_allreduce = elastic_controller.elastic_run(train_one_batch)
        loss = elastic_allreduce(model, optimizer, features, labels)
        step = optimizer.iterations.numpy()
        if step % 5 == 0:
            logger.info("step = {}, loss = {}".format(step, loss))


def train_one_batch(model, optimizer, features, labels):
    with tf.GradientTape() as tape:
        outputs = model.call(features, training=True)
        loss = tf.reduce_mean(
            input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=outputs, labels=tf.reshape(labels, [-1])
            )
        )
    tape = hvd.DistributedGradientTape(tape)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


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
    return features, tf.cast(r["label"], tf.int32)
