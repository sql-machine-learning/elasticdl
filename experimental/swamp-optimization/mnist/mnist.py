from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import sys
import pickle
import threading
import queue


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def trainer(args, up, down):
    kwargs = {}
    data_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data',  # cache data to the current directory.
                       train=True,  # use the training data also for dev.
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size,
        shuffle=True,        # each trainer might have different order
        **kwargs)
    model = Net()               # trainer-local model.
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)

    # model.train()
    score = float("inf")
    step = 0

    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if step < args.free_trial_steps:
                step = step + 1
            else:
                if loss.data < score:
                    score = loss.data
                    if up != None:
                        up.put(pickle.dumps(
                            {"model": model.state_dict(), "loss": loss.data}))
                else:
                    if down != None:
                        m = pickle.loads(down.get())
                        model.load_state_dict(m["model"])
                        score = m["loss"]
                step = 0
            if batch_idx % args.free_trial_steps == 0:
                print(loss)
    print("trainer done enough epochs")


def ps(up, down):
    model = Net()
    score = float("inf")
    updates = 0
    while updates < 50:
        # In the case that any trainer pulls.
        down.put_nowait(pickle.dumps(
            {"model": model.state_dict(), "loss": score}))

        # In the case that any trainer pushes.
        try:
            d = up.get_nowait()
        except queue.Empty:
            continue
        updates = updates + 1
        d = pickle.loads(d)
        if d["loss"] < score:
            model.load_state_dict(d["model"])
            score = d["loss"]
            print(score)
        print("updates", updates)
    print("ps done with enough updates")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--free-trial-steps', type=int, default=10, metavar='N',
                        help='how many batches to wait before sync up with the ps')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    up = queue.Queue()
    down = queue.Queue()
    for t in range(1):
        threading.Thread(target=trainer, args=(args, up, down,)).start()
    ps(up, down)
#    trainer(args, None, None)

    print("main done")


if __name__ == '__main__':
    main()
