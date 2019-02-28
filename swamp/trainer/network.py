import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import subprocess


class MNISTNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def loss_func():
    return nn.CrossEntropyLoss()

def optimizer_func(model_params, lr, momentum, weight_decay):
    return optim.SGD(
        model_params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay)

def preprocess_data():
    status, result = subprocess.getstatusoutput('tar -xf data.tar.gz')
    print('data preprocess status: ' + str(status))
    print('data preprocess output: ' + result)

def prepare_training_dataset():
    return datasets.MNIST('./data',
                          train=True,
                          download=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ]))

def prepare_validation_dataset():
    return datasets.MNIST('./data',
                          train=False,
                          download=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ]))

