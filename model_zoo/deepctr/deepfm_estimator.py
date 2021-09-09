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

import argparse
import csv
import os

import tensorflow as tf
from deepctr.estimator.models import DeepFMEstimator

from elasticai_api.common.data_shard_service import DataShardService
from elasticai_api.tensorflow.hooks import ElasticDataShardReportHook

tf.logging.set_verbosity(tf.logging.INFO)


def read_csv(file_path):
    rows = []
    with open(file_path) as csvfile:
        spamreader = csv.reader(csvfile)
        for i, row in enumerate(spamreader):
            if i > 0:
                row_values = []
                row_values.append(int(row[0]))
                for i in range(1, 14):
                    value = row[i] if row[i] else 0
                    row_values.append(float(value))
                row_values.extend(row[14:])
                rows.append(row_values)
    return rows


def train_generator(data_path, shard_service):
    rows = read_csv(data_path)
    while True:
        # Read samples by the range of the shard from
        # the data shard serice.
        shard = shard_service.fetch_shard()
        if not shard:
            break
        for i in range(shard.start, shard.end):
            yield tuple(rows[i])


def eval_generator(data_path):
    rows = read_csv(data_path)
    for row in rows:
        yield tuple(row)


def input_fn(sample_generator, batch_size, dense_features, sparse_features):
    output_types = tuple(
        [tf.int32]
        + [tf.float32 for i in dense_features]
        + [tf.string for i in sparse_features]
    )
    dataset = tf.data.Dataset.from_generator(
        sample_generator, output_types=output_types,
    )
    dataset = dataset.shuffle(100).batch(batch_size)
    values = dataset.make_one_shot_iterator().get_next()

    label_value = values[0]
    feature_values = {}
    feature_index = 1
    for feature in dense_features:
        feature_values[feature] = values[feature_index]
        feature_index += 1

    for feature in sparse_features:
        feature_values[feature] = values[feature_index]
        feature_index += 1
    return feature_values, label_value


def arg_parser():
    parser = argparse.ArgumentParser(description="Process training parameters")
    parser.add_argument("--training_data", type=str, required=True)
    parser.add_argument(
        "--validation_data", type=str, default="", required=False
    )
    return parser


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()

    training_data = args.training_data
    validation_data = args.validation_data

    model_dir = "/data/ckpts/"
    os.makedirs(model_dir, exist_ok=True)

    sparse_features = ["C" + str(i) for i in range(1, 27)]
    dense_features = ["I" + str(i) for i in range(1, 14)]

    dnn_feature_columns = []
    linear_feature_columns = []

    for i, feat in enumerate(sparse_features):
        dnn_feature_columns.append(
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_hash_bucket(
                    feat, 1000
                ),
                4,
            )
        )
        linear_feature_columns.append(
            tf.feature_column.categorical_column_with_hash_bucket(feat, 1000)
        )
    for feat in dense_features:
        dnn_feature_columns.append(tf.feature_column.numeric_column(feat))
        linear_feature_columns.append(tf.feature_column.numeric_column(feat))

    batch_size = 64

    config = tf.estimator.RunConfig(
        model_dir=model_dir, save_checkpoints_steps=100, keep_checkpoint_max=3
    )
    model = DeepFMEstimator(
        linear_feature_columns,
        dnn_feature_columns,
        task="binary",
        config=config,
    )

    # Create a data shard service which can split the dataset
    # into shards.
    rows = read_csv(training_data)
    training_data_shard_svc = DataShardService(
        batch_size=batch_size,
        num_epochs=100,
        dataset_size=len(rows),
        num_minibatches_per_shard=1,
        dataset_name="iris_training_data",
    )

    def train_input_fn():
        return input_fn(
            lambda: train_generator(training_data, training_data_shard_svc),
            batch_size,
            dense_features,
            sparse_features,
        )

    def eval_input_fn():
        return input_fn(
            lambda: eval_generator(validation_data),
            batch_size,
            dense_features,
            sparse_features,
        )

    hooks = [
        ElasticDataShardReportHook(training_data_shard_svc),
    ]
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, hooks=hooks)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
