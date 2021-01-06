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
We can use the following command to submit a training job with the script.
elasticdl train \
  --image_name=elasticdl:dev_allreduce  \
  --training_data=/data/mnist/train/images.csv \
  --job_command="python -m model_zoo.mnist.mnist_pytorch_custom_dataset" \
  --num_epochs=1 \
  --num_minibatches_per_task=2 \
  --minibatch_size=64 \
  --num_workers=2 \
  --worker_pod_priority=0.5 \
  --master_resource_request="cpu=0.2,memory=1024Mi" \
  --master_resource_limit="cpu=1,memory=2048Mi" \
  --worker_resource_request="cpu=0.3,memory=1024Mi" \
  --worker_resource_limit="cpu=1,memory=2048Mi" \
  --envs="USE_TORCH=true" \
  --job_name=mnist-allreduce \
  --image_pull_policy=Never \
  --distribution_strategy=AllreduceStrategy \
  --need_elasticdl_job_service=false \
"""

import argparse
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

from elasticai_api.pytorch.controller import create_elastic_controller
from elasticai_api.pytorch.optimizer import DistributedOptimizer


class ElasticDataset(Dataset):
    def __init__(self, data_shard_service, images):
        self.data_shard_service = data_shard_service
        self.images = images

    def __len__(self):
        """Set the maxsize because the size of dataset is not fixed
        when using dynamic sharding"""
        return sys.maxsize

    def __getitem__(self, index):
        index = self.data_shard_service.fetch_record_index()
        image_path, label = self.images[index]
        image = cv2.imread(image_path)
        image = np.array(image / 255.0, np.float32)
        image = image.reshape(3, 28, 28)
        return image, label


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
    training_data = torchvision.datasets.ImageFolder(args.training_data)
    allreduce_controller = create_elastic_controller(
        batch_size=args.batch_size,
        dataset_size=len(training_data.imgs),
        num_epochs=args.num_epochs,
        shuffle=True,
    )
    dataset = ElasticDataset(
        allreduce_controller.data_shard_service, training_data.imgs
    )
    data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, num_workers=2
    )
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    optimizer = DistributedOptimizer(optimizer, fixed_global_batch_size=True)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

    # Set the model and optimizer to broadcast.
    allreduce_controller.set_broadcast_model(model)
    allreduce_controller.set_broadcast_optimizer(optimizer)
    model.train()
    epoch = 0
    with allreduce_controller.scope():
        for batch_idx, (data, target) in enumerate(data_loader):

            new_epoch = int(
                allreduce_controller.global_completed_batch_num / 100
            )
            if new_epoch > epoch:
                epoch = new_epoch
                scheduler.step()

            target = target.type(torch.LongTensor)

            # Use the elastic function to wrap the training function with
            # a batch.
            elastic_train_one_batch = allreduce_controller.elastic_run(
                train_one_batch
            )
            loss = elastic_train_one_batch(model, optimizer, data, target)
            print("loss = {}, step = {}".format(loss, batch_idx))


def train_one_batch(model, optimizer, data, target):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    return loss


def arg_parser():
    parser = argparse.ArgumentParser(description="Process training parameters")
    parser.add_argument("--batch_size", type=int, default=64, required=False)
    parser.add_argument("--num_epochs", type=int, default=1, required=False)
    parser.add_argument(
        "--learning_rate", type=float, default=0.1, required=False
    )
    parser.add_argument("--training_data", type=str, required=True)
    parser.add_argument(
        "--validation_data", type=str, default="", required=False
    )
    return parser


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    train(args)
