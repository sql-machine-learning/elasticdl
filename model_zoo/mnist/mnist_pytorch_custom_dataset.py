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
Download the mnist dataset from
https://s3.amazonaws.com/fast-ai-imageclas/mnist_png.tgz
and then untar it into ${data_store_dir}. Using minikube, we can use the
following command to submit a training job with the script.

elasticdl train \
  --image_name=elasticdl:pt_mnist_allreduce  \
  --job_command="python -m model_zoo.mnist.mnist_pytorch_custom_dataset \
      --training_data=/local_data/mnist_png/training \
      --validation_data=/local_data/mnist_png/testing" \
  --num_minibatches_per_task=2 \
  --num_workers=2 \
  --worker_pod_priority=0.5 \
  --master_resource_request="cpu=0.2,memory=1024Mi" \
  --master_resource_limit="cpu=1,memory=2048Mi" \
  --worker_resource_request="cpu=0.3,memory=1024Mi" \
  --worker_resource_limit="cpu=1,memory=2048Mi" \
  --envs="USE_TORCH=true,HOROVOD_GLOO_TIMEOUT_SECONDS=60,PYTHONUNBUFFERED=0" \
  --job_name=test-mnist-allreduce \
  --image_pull_policy=Never \
  --volume="host_path=${data_store_dir},mount_path=/local_data" \
  --custom_training_loop=true \
  --distribution_strategy=AllreduceStrategy \
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
    def __init__(self, images, data_shard_service=None):
        """The dataset supports elastic training.

        Args:
            images: A list with tuples like (image_path, label_index).
            For example, we can use `torchvision.datasets.ImageFolder`
            to get the list.
            data_shard_service: If we want to use elastic training, we
            need to use the `data_shard_service` of the elastic controller
            in elasticai_api.
        """
        self.data_shard_service = data_shard_service
        self._images = images

    def __len__(self):
        if self.data_shard_service:
            # Set the maxsize because the size of dataset is not fixed
            # when using dynamic sharding
            return sys.maxsize
        else:
            return len(self._images)

    def __getitem__(self, index):
        if self.data_shard_service:
            index = self.data_shard_service.fetch_record_index()
            return self.read_image(index)
        else:
            return self.read_image(index)

    def read_image(self, index):
        image_path, label = self._images[index]
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
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_data = torchvision.datasets.ImageFolder(args.training_data)
    test_data = torchvision.datasets.ImageFolder(args.validation_data)
    batch_num_per_epoch = int(len(train_data.imgs) / args.batch_size)

    allreduce_controller = create_elastic_controller(
        batch_size=args.batch_size,
        dataset_size=len(train_data.imgs),
        num_epochs=args.num_epochs,
        shuffle=True,
    )
    train_dataset = ElasticDataset(
        train_data.imgs, allreduce_controller.data_shard_service
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, num_workers=2
    )

    test_dataset = ElasticDataset(test_data.imgs)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, num_workers=2
    )

    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    optimizer = DistributedOptimizer(optimizer, fixed_global_batch_size=True)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

    # Set the model and optimizer to broadcast.
    allreduce_controller.set_broadcast_model(model)
    allreduce_controller.set_broadcast_optimizer(optimizer)
    epoch = 0
    # Use the elastic function to wrap the training function with a batch.
    elastic_train_one_batch = allreduce_controller.elastic_run(train_one_batch)
    with allreduce_controller.scope():
        for batch_idx, (data, target) in enumerate(train_loader):
            model.train()
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)
            loss = elastic_train_one_batch(model, optimizer, data, target)
            print("loss = {}, step = {}".format(loss, batch_idx))
            new_epoch = int(
                allreduce_controller.global_completed_batch_num
                / batch_num_per_epoch
            )
            if new_epoch > epoch:
                epoch = new_epoch
                # Set epoch of the scheduler
                scheduler.last_epoch = epoch - 1
                scheduler.step()
                test(model, device, test_loader)


def train_one_batch(model, optimizer, data, target):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    return loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def arg_parser():
    parser = argparse.ArgumentParser(description="Process training parameters")
    parser.add_argument("--batch_size", type=int, default=64, required=False)
    parser.add_argument("--num_epochs", type=int, default=1, required=False)
    parser.add_argument(
        "--learning_rate", type=float, default=0.1, required=False
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disable CUDA training",
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
