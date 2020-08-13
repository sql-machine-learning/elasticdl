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

import numpy as np
import PIL.Image
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from elasticdl.python.common.constants import Mode


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 3 * 3, 625)
        self.fc2 = nn.Linear(625, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def loss(labels, predictions):
    labels = labels.long()  # labels.dtype must be Long/int64 for CrossEntropyLoss()
    loss_func = nn.CrossEntropyLoss()
    return loss_func(predictions, labels)


# def optimizer(model,lr=0.001):
#     return torch.optim.Adam(model.parameters(), lr)
def optimizer(lr=0.01):
    return tf.optimizers.SGD(lr)


def dataset_fn(dataset, mode, _):
    def _parse_data(record):
        if mode == Mode.PREDICTION:
            feature_description = {
                "image": tf.io.FixedLenFeature([28, 28], tf.float32)
            }
        else:
            feature_description = {
                "image": tf.io.FixedLenFeature([28, 28], tf.float32),
                "label": tf.io.FixedLenFeature([1], tf.int64),
            }
        r = tf.io.parse_single_example(record, feature_description)
        features = {
            "image": tf.math.divide(tf.cast(r["image"], tf.float32), 255.0)
        }
        if mode == Mode.PREDICTION:
            return features
        else:
            return features, tf.cast(r["label"], tf.int32)

    dataset = dataset.map(_parse_data)

    if mode == Mode.TRAINING:
        dataset = dataset.shuffle(buffer_size=1024)
    return dataset


def eval_metrics_fn():
    return {
        "accuracy": lambda labels, predictions: tf.equal(
            tf.argmax(predictions, 1, output_type=tf.int32),
            tf.cast(tf.reshape(labels, [-1]), tf.int32),
        )
    }


class CustomDataset(torch.utils.data.IterableDataset):
    def __init__(self, data):
        self.data_source = data

    def __iter__(self):
        return iter(self.data_source)


def _dataset_pytorch(dataset, batch_size):
    """
    _dataset_fn() builds dataset for TensorFlow
    this func transforms dataset and set DataLoader for PyTorch
    TODO: rewrite _dataset_fn() by IterableDataSet for PyTorch
    """
    dataset_list = []
    for data_enum in list(dataset.as_numpy_iterator()):
        shape_0 = data_enum[0].shape[0]
        for i in range(shape_0):
            dataset_list.append((data_enum[0][i:i + 1, ...], data_enum[1][i:i + 1, ...]))

    iterable_dataset = CustomDataset(dataset_list)
    dataloader = DataLoader(dataset=iterable_dataset, batch_size=batch_size)
    return dataloader