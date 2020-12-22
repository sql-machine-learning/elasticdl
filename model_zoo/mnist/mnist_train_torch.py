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

import horovod.torch as hvd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from elasticai_api.pytorch.optimizer import DistributedOptimizer
from elasticdl.python.common.constants import Mode
from elasticdl.python.common.log_utils import default_logger as logger


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(dataset, elastic_controller):
    """ The function to run the training loop.

    Args:
        dataset: The dataset is provided by ElasticDL for the elastic training.
        Now, the dataset if tf.data.Dataset and we need to convert
        the data in dataset to torch.tensor. Later, ElasticDL will
        pass a torch.utils.data.DataLoader.
        elastic_controller: The controller for elastic training.
    """
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # op must be sum to keep the batch size fixed
    optimizer = DistributedOptimizer(
        optimizer, op=hvd.Average, fixed_global_batch_size=True,
    )

    # Set the model and optimizer to broadcast.
    elastic_controller.set_broadcast_model(model)
    elastic_controller.set_broadcast_optimizer(optimizer)
    model.train()

    # Use the elastic function to wrap the training function with a batch.
    elastic_train_one_step = elastic_controller.elastic_run(train_one_batch)
    for batch_idx, (data, target) in enumerate(dataset):
        # Convert tf.tensor to torch.tensor.
        target = tf.reshape(target, [-1])
        data = tf.expand_dims(data, axis=1)
        data = torch.from_numpy(data.numpy())
        target = torch.from_numpy(target.numpy())

        target = target.type(torch.LongTensor)

        loss = elastic_train_one_step(
            batch_idx, model, optimizer, data, target
        )
        logger.info("loss = {}, batch_index = {}".format(loss, batch_idx))


def train_one_batch(batch_index, model, optimizer, data, target):
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
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
    return features["image"], tf.cast(r["label"], tf.int32)
