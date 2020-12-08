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
import linecache
import os

import cv2
import horovod.torch as hvd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

from elasticdl.python.allreduce.pytorch_controller import (
    create_elastic_controller,
)
from elasticdl.python.common.log_utils import default_logger as logger


def read_images(shard):
    """
    shard.name is a CSV file path, like "/data/mnist/train/images.csv"
    shard.start and shard.end is the line number of the CSV file.

    The content of the CSV like:
        5/img_26887.jpg,5
        5/img_9954.jpg,5
        5/img_29578.jpg,5
        5/img_7979.jpg,5

    Each line of the CSV contains the relative path and label.
    """
    filename = shard.name.split(":")[0]
    records = linecache.getlines(filename)[shard.start : shard.end]
    file_path = os.path.dirname(filename)

    images = []
    for record in records:
        image_path = os.path.join(file_path, record.split(",")[0])
        label = int(record.split(",")[1])
        image = cv2.imread(image_path)
        image = np.array(image / 255.0, np.float32)
        image = image.reshape(3, 28, 28)
        images.append((image, label))
    return images


class ImageDataset(IterableDataset):
    def __init__(self, data_shard_service, shuffle=False):
        self.data_shard_service = data_shard_service
        self._shuffle = shuffle

    def __iter__(self):
        while True:
            shard = self.data_shard_service.fetch_shard()
            if shard:
                images = read_images(shard)
                if self._shuffle:
                    np.random.shuffle(images)
                for image in images:
                    yield image
            else:
                break


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
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


def train(args):
    """ The function to run the training loop.

    Args:
        dataset: The dataset is provided by ElasticDL for the elastic training.
        Now, the dataset if tf.data.Dataset and we need to convert
        the data in dataset to torch.tensor. Later, ElasticDL will
        pass a torch.utils.data.DataLoader.
        elastic_controller: The controller for elastic training.
    """
    allreduce_controller = create_elastic_controller(
        batch_size=args.batch_size
    )
    dataset = ImageDataset(
        allreduce_controller.data_shard_service, shuffle=True
    )
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size)
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    optimizer = hvd.DistributedOptimizer(optimizer)

    # Set the model and optimizer to broadcast.
    allreduce_controller.set_broadcast_model(model)
    allreduce_controller.set_broadcast_optimizer(optimizer)
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        target = target.type(torch.LongTensor)

        # Use the elastic function to wrap the training function with a batch.
        elastic_train_one_batch = allreduce_controller.elastic_run(
            train_one_batch
        )
        loss = elastic_train_one_batch(model, optimizer, data, target)
        logger.info("loss = {}, step = {}".format(loss, batch_idx))


def train_one_batch(model, optimizer, data, target):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    return loss


def arg_parser():
    parser = argparse.ArgumentParser(description="Process training parameters")
    parser.add_argument(
        "batch_size", type=int, default=64,
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.1,
    )
    return parser


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    train(args)
