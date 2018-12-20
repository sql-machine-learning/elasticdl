from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import sys
import pickle
import threading
import queue
import time
import gc
from matplotlib import pyplot as plot
import random


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


class Trainer(object):

    def __init__(self, tid, args, up, down):
        self._tid = tid
        self._args = args
        self._up = up
        self._down = down
        self._time_costs = []
        self._losses = []

    def train(self):
        kwargs = {}
        data_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data',  # cache data to the current directory.
                           train=True,  # use the training data also for dev.
                           download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=self._args.batch_size,
            shuffle=True,        # each trainer might have different order
            **kwargs)

        model = Net()               # trainer-local model.
        optimizer = optim.SGD(model.parameters(), lr=self._args.lr,
                              momentum=self._args.momentum)

        # model.train()
        score = float("inf")
        step = 0

        start_time = time.time()
        for epoch in range(self._args.epochs):
            for batch_idx, (data, target) in enumerate(data_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)

                if step < self._args.free_trial_steps:
                    loss.backward()
                    optimizer.step()
                    step = step + 1
                else:
                    if loss.data < score:
                        score = loss.data
                        if self._up != None:
                            self._up.put(pickle.dumps(
                                {"model": model.state_dict(), "opt": optimizer.state_dict(), "loss": loss.data}))
                    else:
                        if self._down != None and random.random() < self._args.pull_probability:
                            m = pickle.loads(self._down.get())
                            model.load_state_dict(m["model"])
                            optimizer.load_state_dict(m["opt"])
                            score = m["loss"]
                    step = 0
                gc.collect()

                if self._args.loss_file is not None:
                    self._time_costs.append(round(time.time() - start_time))
                    self._losses.append(round(loss.item(), 4))

                if batch_idx % self._args.log_interval == 0:
                    print("Current trainer id: %i, epoch: %i, batch id: %i" %
                          (self._tid, epoch, batch_idx))
            print("trainer %i done epoch %i" % (self._tid, epoch))

    def time_costs(self):
        return self._time_costs

    def losses(self):
        return self._losses

    def tid(self):
        return self._tid


class Ps(object):

    def __init__(self, args, up, down):
        self._args = args
        self._up = up
        self._down = down
        self._time_costs = []
        self._losses = []
        self._exit = False

    def run(self):
        model_and_score = None
        score = float("inf")
        updates = 0
        batch_size = self._args.validate_batch_size
        max_batch = self._args.validate_max_batch

        validate_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data',
                           train=False,
                           download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size,
            shuffle=True)  # shuffle for random test

        start_time = time.time()
        while not self._exit:
            # In the case that any trainer pulls.
            if model_and_score != None:
                self._down.put_nowait(model_and_score)

            # In the case that any trainer pushes.
            try:
                d = self._up.get(timeout=1.0)
            except queue.Empty:
                continue

            s = pickle.loads(d)["loss"]

            # Restore uploaded model
            state_dict = pickle.loads(d)["model"]
            model = Net()
            model.load_state_dict(state_dict)

            if s < score:
                # Model double check
                double_check_loss = self.validate(
                    model, validate_loader, batch_size, max_batch)
                if double_check_loss < score:
                    model_and_score = d
                    score = s
                    updates = updates + 1

                    if self._args.loss_file is not None:
                        self._time_costs.append(
                            round(time.time() - start_time))
                        self._losses.append(round(s.item(), 4))

    def validate(self, model, data_loader, batch_size, max_batch):
        eval_loss = 0
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(data_loader):
                if batch_idx < max_batch:
                    out = model(batch_x)
                    loss = F.nll_loss(out, batch_y)
                    eval_loss += loss.data.item()
                else:
                    break
        loss_val = eval_loss / max_batch
        return loss_val

    def time_costs(self):
        return self._time_costs

    def losses(self):
        return self._losses

    def join(self):
        self._exit = True


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
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
    parser.add_argument('--validate_batch_size', default=64,
                        help='batch size for validation dataset in ps')
    parser.add_argument('--validate_max_batch', default=5,
                        help='max batch for validate model in ps')
    parser.add_argument('--loss-file', default='loss.png',
                        help='the name of loss figure file')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--pull-probability', type=float, default=0,
                        help='the probability of trainer pulling from ps')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    up = queue.Queue()
    down = queue.Queue()

    ps = Ps(args, up, down)
    ps_thread = threading.Thread(target=ps.run, name='ps')
    ps_thread.start()

    trainers = []
    trainer_threads = []
    for t in range(2):
        trainer = Trainer(t, args, up, down)
        trainer_thread = threading.Thread(
            target=trainer.train, name=('trainer-' + str(t)))
        trainer_thread.start()
        trainers.append(trainer)
        trainer_threads.append(trainer_thread)

    for thread in trainer_threads:
        thread.join()

    ps.join()

    if args.loss_file is not None:
        print("Write image to ", args.loss_file)
        plot.xlabel('timestamp')
        plot.ylabel('loss')
        plot.title('swamp training for mnist data')
        for trainer in trainers:
            plot.plot(trainer.time_costs(), trainer.losses(),
                      label='trainer-' + str(trainer.tid()))
        plot.plot(ps.time_costs(), ps.losses(), label='ps')
        plot.legend(loc=7)
        plot.savefig(args.loss_file)


if __name__ == '__main__':
    main()
