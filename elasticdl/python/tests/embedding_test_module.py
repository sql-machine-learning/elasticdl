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

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Flatten

from elasticdl.python.elasticdl.layers.embedding import Embedding


class EdlEmbeddingModel(tf.keras.Model):
    def __init__(self, output_dim=16, weights=None):
        """
        Arguments:
            output_dim: An Integer. It is the output dimension of embedding
                layers in `EdlEmbeddingModel`.
            weights: A numpy ndarray list. If `weights` is not None, dense
                layer initializes its weights using `weights`.
        """
        super(EdlEmbeddingModel, self).__init__(name="EdlEmbeddingModel")
        self.output_dim = output_dim
        if weights:
            if len(weights) != 2:
                raise ValueError(
                    "EdlEmbeddingModel constructor receives weights with "
                    "length %d, expected %d" % (len(weights), 2)
                )

        self.embedding_1 = Embedding(output_dim)
        self.embedding_2 = Embedding(output_dim)
        self.concat = Concatenate()
        self.dense = Dense(1, weights=weights)
        self.flatten = Flatten()

    def call(self, inputs, training=False):
        x = self.concat(
            [
                self.embedding_1(inputs["f1"]),
                self.embedding_1(inputs["f2"]),
                self.embedding_2(inputs["f3"]),
            ]
        )
        x = self.flatten(x)
        x = self.dense(x)
        return x


# The model structure of KerasEmbeddingModel should keep same with
# EdlEmbeddingModel.
class KerasEmbeddingModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim=16, weights=None):
        """
        Arguments:
            input_dim: An Integer. It is the input dimension of embedding
                layers in `KerasEmbeddingModel`.
            output_dim: An Integer. It is the output dimension of embedding
                layers in `KerasEmbeddingModel`.
            weights: A numpy ndarray list. Unless `weights` is None, embedding
                layer and dense layer initialize their weights using `weights`.
        """
        super(KerasEmbeddingModel, self).__init__(name="KerasEmbeddingModel")
        self.output_dim = output_dim
        if weights:
            weight_1 = [weights[0]]
            weight_2 = [weights[1]]
            linear_weights = weights[2:]
        else:
            weight_1, weight_2, linear_weights = None, None, None
        self.embedding_1 = tf.keras.layers.Embedding(
            input_dim, output_dim, weights=weight_1
        )
        self.embedding_2 = tf.keras.layers.Embedding(
            input_dim, output_dim, weights=weight_2
        )
        self.concat = Concatenate()
        self.dense = Dense(1, weights=linear_weights)
        self.flatten = Flatten()

    def call(self, inputs, training=False):
        x = self.concat(
            [
                self.embedding_1(inputs["f1"]),
                self.embedding_1(inputs["f2"]),
                self.embedding_2(inputs["f3"]),
            ]
        )
        x = self.flatten(x)
        x = self.dense(x)
        return x


def loss(labels, predictions):
    return tf.reduce_mean(tf.square(predictions - labels))


def dataset_fn(dataset, mode, metadata):
    def _parse_data(record):
        feature_description = {
            "f1": tf.io.FixedLenFeature([1], tf.int64),
            "f2": tf.io.FixedLenFeature([1], tf.int64),
            "f3": tf.io.FixedLenFeature([1], tf.int64),
            "label": tf.io.FixedLenFeature([1], tf.int64),
        }
        r = tf.io.parse_single_example(record, feature_description)
        return {"f1": r["f1"], "f2": r["f2"], "f3": r["f3"]}, r["label"]

    dataset = dataset.map(_parse_data)
    return dataset


def optimizer(lr=0.1):
    return tf.optimizers.SGD(lr)


def eval_metrics_fn(predictions, labels):
    return {"mse": tf.reduce_mean(tf.square(predictions - labels))}
