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
import torchvision
import torch.nn.functional as F

from elasticdl.python.common.constants import Mode

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,padding = 1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,3,padding = 1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64,128,3,padding = 1)
        self.pool3 = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(128*3*3,625)
        self.fc2 = nn.Linear(625,10)
          
    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1,128*3*3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def prepare_data_for_a_single_file(file_object, filename):
    """
    :param filename: training data file name
    :param file_object: a file object associated with filename
    """
    label = int(filename.split("/")[-2])
    image = PIL.Image.open(file_object)
    numpy_image = np.array(image)
    example_dict = {
        "image": tf.train.Feature(
            float_list=tf.train.FloatList(value=numpy_image.flatten())
        ),
        "label": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[label])
        ),
    }
    example = tf.train.Example(
        features=tf.train.Features(feature=example_dict)
    )
    return example.SerializeToString()


def loss(labels, predictions):
    # loss_func = nn.CrossEntropyLoss()
    loss_func = nn.NLLLoss()
    print("loss_func:")
    print("labels:",labels.dtype)
    print("predictions:",predictions.dtype)
    loss = loss_func(predictions, labels)
    return loss

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

def dataset_pytorch(dataset, minibatch_size):
    dataset = list(dataset.as_numpy_iterator())
    iterable_dataset = CustomDataset(dataset)
    dataloader = DataLoader(dataset=iterable_dataset, batch_size=minibatch_size)
    return dataloader