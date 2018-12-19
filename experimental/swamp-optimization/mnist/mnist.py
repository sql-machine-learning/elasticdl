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
import gc


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

            if step < args.free_trial_steps:
                loss.backward()
                optimizer.step()
                step = step + 1
            else:
                if loss.data < score:
                    score = loss.data
                    if up != None:
                        up.put(pickle.dumps(
                            {"model": model.state_dict(), "opt": optimizer.state_dict(), "loss": loss.data}))
                else:
                    if down != None:
                        m = pickle.loads(down.get())
                        model.load_state_dict(m["model"])
                        optimizer.load_state_dict(m["opt"])
                        score = m["loss"]
                step = 0
        print("trainer done epoch", epoch)


def ps(args, up, down):
    model_and_score = None
    score = float("inf")
    updates = 0
    batch_size = args.validate_batch_size
    max_batch = args.validate_max_batch

    validate_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data',
                       train=False,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
                       batch_size=batch_size,
                       shuffle=True) # shuffle for random test

    while updates < 500:
        # In the case that any trainer pulls.
        if model_and_score != None:
            down.put_nowait(model_and_score)

        # In the case that any trainer pushes.
        try:
            d = up.get(timeout=1.0)
        except queue.Empty:
            continue
        s = pickle.loads(d)["loss"]

        # Restore uploaded model
        state_dict = pickle.loads(d)["model"]
        model = Net()
        model.load_state_dict(state_dict)
        
        if s < score:
            # Model double check
            double_check_loss = validate(model, validate_loader, batch_size, max_batch)
            if double_check_loss < score:
                model_and_score = d
                score = s
                updates = updates + 1
                print("updated", updates, score.data.item(), double_check_loss)
        gc.collect()

def validate(model, data_loader, batch_size, max_batch):
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
    parser.add_argument('--validate_batch_size', default=64,
                        help='batch size for validation dataset in ps')
    parser.add_argument('--validate_max_batch', default=5,
                        help='max batch for validate model in ps')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    up = queue.Queue()
    down = queue.Queue()
    for t in range(4):
        threading.Thread(target=trainer, args=(args, up, down,)).start()
    ps(args, up, down)


if __name__ == '__main__':
    main()
