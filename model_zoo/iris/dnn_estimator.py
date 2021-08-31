# Copyright 2021 The ElasticDL Authors. All rights reserved.
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

import csv
import logging
import os

import tensorflow as tf

from elasticai_api.common.data_shard_service import build_data_shard_service

tf.logging.set_verbosity(tf.logging.INFO)

CATEGORY_CODE = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
DATASET_DIR = "/data/iris.data"


def read_csv(file_path):
    rows = []
    with open(file_path) as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            rows.append(row)
    return rows


class ElasticDataShardReportHook(tf.train.SessionRunHook):
    def __init__(self, data_shard_service) -> None:
        self._data_shard_service = data_shard_service

    def after_run(self, run_context, run_values):
        try:
            self._data_shard_service.report_batch_done()
        except Exception as ex:
            logging.info("elastic_ai: report batch done failed: %s", ex)


class LocalStepHook(tf.train.SessionRunHook):
    """Logs loss and runtime."""

    def begin(self):
        self._step = 0

    def after_run(self, run_context, run_values):
        self._step += 1
        if self._step % 100 == 0:
            logging.info(
                "step = {}, value = {}".format(self._step, run_values)
            )


def model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params["feature_columns"])

    for units in params["hidden_units"]:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    logits = tf.layers.dense(net, params["n_classes"], activation=None)

    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "classes": predicted_classes[:, tf.newaxis],
            "probs": tf.nn.softmax(logits),
            "logits": logits,
        }
        export_outputs = {
            "prediction": tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs
        )

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step()
        )
        logging_hook = tf.train.LoggingTensorHook(
            {"loss": loss}, every_n_iter=10
        )
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op, training_hooks=[logging_hook]
        )

    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predicted_classes, name="acc"
    )
    metrics = {"accuracy": accuracy}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics
        )


def train_generator(shard_service):
    rows = read_csv(DATASET_DIR)
    while True:
        # Read samples by the range of the shard from
        # the data shard serice.
        shard = shard_service.fetch_shard()
        if not shard:
            break
        for i in range(shard.start, shard.end):
            label = CATEGORY_CODE[rows[i][-1]]
            yield rows[i][0:-1], [label]


def eval_generator():
    rows = read_csv(DATASET_DIR)
    for row in rows:
        label = CATEGORY_CODE[row[-1]]
        yield row[0:-1], [label]


def input_fn(sample_generator, batch_size):
    dataset = tf.data.Dataset.from_generator(
        sample_generator,
        output_types=(tf.float32, tf.int32),
        output_shapes=(4, 1),
    )
    dataset = dataset.shuffle(100).batch(batch_size)
    feature_values, label_values = dataset.make_one_shot_iterator().get_next()
    features = {"x": feature_values}
    return features, label_values


if __name__ == "__main__":
    model_dir = "/data/ckpts/"
    batch_size = 64
    feature_columns = [
        tf.feature_column.numeric_column(key="x", shape=(4,), dtype=tf.float32)
    ]
    os.makedirs(model_dir, exist_ok=True)

    config = tf.estimator.RunConfig(
        model_dir=model_dir, save_checkpoints_steps=300, keep_checkpoint_max=5
    )
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params={
            "hidden_units": [8, 4],
            "n_classes": 3,
            "feature_columns": feature_columns,
        },
    )

    # Create a data shard service which can split the dataset
    # into shards.
    rows = read_csv(DATASET_DIR)
    training_data_shard_svc = build_data_shard_service(
        batch_size=batch_size,
        num_epochs=100,
        dataset_size=len(rows),
        num_minibatches_per_shard=1,
        dataset_name="iris_training_data",
    )

    hooks = [
        LocalStepHook(),
        ElasticDataShardReportHook(training_data_shard_svc),
    ]

    def train_input_fn():
        return input_fn(
            lambda: train_generator(training_data_shard_svc), batch_size
        )

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, hooks=hooks,)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(eval_generator, batch_size)
    )
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
