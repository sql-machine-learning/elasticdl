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
from multiprocessing import Process, Queue, Manager
from ctypes import py_object
import queue
import time
import gc
from matplotlib import pyplot as plot
import random
import os
import shutil


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class TrainedModel(object):
    ''' Model uploaded to PS by trainers
    '''

    def __init__(self, model_state, loss=float("inf"), version=1):
        self.model_state = model_state
        self.loss = loss
        self.version = version


class Metrics(object):
    def __init__(self, loss, accuracy, timestamp=1):
        self.loss = loss
        self.accuracy = accuracy
        self.timestamp = timestamp


class Trainer(object):

    def __init__(
            self,
            job_dir,
            tid,
            args,
            trained_model_wrapper,
            up):
        """ Initialize the Trainer.

        Arguments:
          job_dir: Path for storing experimental data.
          tid: The unique identifier of the trainer.
          args: Runtime arguments.
          trained_model_wrapper: The info(eg. state, loss) of the best uploaded
                                 model at present model shared by PS and trainer
                                 which is managed by Manager.
          up: A shared Queue for trainer upload model to PS.
        """
        # Create model data dir for this trainer.
        self._model_dir = '{}/trainer_{}'.format(job_dir, tid)
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)

        self.tid = tid
        self._args = args
        self._up = up
        self._start_time = time.time()
        self._model = Net()
        self._optimizer = optim.SGD(self._model.parameters(), lr=self._args.lr,
                                    momentum=self._args.momentum)
        self._score = float("inf")
        self._trained_model_wrapper = trained_model_wrapper

    def train(self):
        data_loader = self._prepare_dataloader(self._args.batch_size)
        step = 0

        # start local training
        for epoch in range(self._args.epochs):
            for batch_idx, (data, target) in enumerate(data_loader):
                self._optimizer.zero_grad()
                output = self._model(data)
                loss = F.nll_loss(output, target)

                if step < self._args.free_trial_steps:
                    loss.backward()
                    self._optimizer.step()
                    step = step + 1
                else:
                    if loss.data < self._score:
                        self._push_model(loss)
                    else:
                        if random.random() < self._args.pull_probability:
                            self._pull_model()
                    step = 0

                gc.collect()
                if batch_idx % self._args.loss_sample_interval == 0:
                    self._dump_model(epoch, batch_idx)
                self._print_progress(epoch, batch_idx)
            print("trainer %i done epoch %i" % (self.tid, epoch))

    def _prepare_dataloader(self, batch_size):
        kwargs = {}
        return torch.utils.data.DataLoader(
            datasets.MNIST('./data',  # cache data to the current directory.
                           train=True,  # use the training data also for dev.
                           download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size,
            shuffle=True,        # each trainer might have different order
            **kwargs)

    def _pull_model(self):
        trained_model = self._trained_model_wrapper.value
        if random.random() < self._args.crossover_probability:
            s_dict = self._model.state_dict()
            for k in trained_model.model_state:
                s_dict[k] = (s_dict[k] + trained_model.model_state[k]) / 2
        else:
            self._model.load_state_dict(trained_model.model_state)
        self._score = trained_model.loss
        self._pulled_losses.append(self._score.data.item())
        self._pull_timestamps.append(timestamps(self._start_time))

    def _push_model(self, loss):
        self._score = loss.data
        if self._up is not None:
            upload_model = TrainedModel(
                self._model.state_dict(), loss.data)
            self._up.put(pickle.dumps(upload_model))

    def _dump_model(self, epoch, batch_idx):
        torch.save(
            self._model.state_dict(),
            '{}/model_params_trainer_{}_epoch_{}_batch_{}_sec_{}.pkl'.format(
                self._model_dir,
                self.tid,
                epoch,
                batch_idx,
                timestamps(
                    self._start_time)))

    def _print_progress(self, epoch, batch_idx):
        if batch_idx % self._args.log_interval == 0:
            print("Current trainer id: %i, epoch: %i, batch id: %i" %
                  (self.tid, epoch, batch_idx))


class PS(object):

    def __init__(
            self,
            job_dir,
            ps_id,
            args,
            trained_model_wrapper,
            up):
        """ Initialize the PS.

        Arguments:
          job_dir: Path for storing experimental data.
          ps_id: The unique identifier for this PS.
          args: Runtime arguments.
          trained_model_wrapper: The info(eg. state, loss) of the best uploaded
                                 model at present shared by PS and trainer which
                                 is managed by Manager.
          up: A shared Queue for trainer upload model to PS.
        """
        # Create model data dir for PS.
        self._model_dir = '{}/ps_{}'.format(job_dir, ps_id)
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)

        self._ps_id = ps_id
        self._args = args
        self._up = up
        self._exit = False
        self._start_time = time.time()
        self._model = Net()
        self._trained_model_wrapper = trained_model_wrapper
        self._score = float("inf")
        self._validate_score = float("inf")

    def run(self):
        updates = 0
        validate_loader = self._prepare_validation_loader()

        while not self._exit:
            # In the case that any trainer pushes.
            try:
                d = self._up.get(timeout=1.0)
            except queue.Empty:
                continue

            # Restore uploaded model
            upload_model = pickle.loads(d)
            self._model.load_state_dict(upload_model.model_state)

            if upload_model.loss < self._score:
                # Model double check
                double_check_loss, accuracy = self._validate(validate_loader)
                if double_check_loss < self._validate_score:
                    self._update_model_wrapper(upload_model)
                    self._validate_score = double_check_loss
                    self._dump_model(upload_model.version)

    def _prepare_validation_loader(self):
        return torch.utils.data.DataLoader(
            datasets.MNIST('./data',
                           train=False,
                           download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=self._args.validate_batch_size,
            shuffle=True)  # shuffle for random test

    def _validate(self, data_loader):
        max_batch = self._args.validate_max_batch
        eval_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(data_loader):
                if batch_idx < max_batch:
                    out = self._model(batch_x)
                    loss = F.nll_loss(out, batch_y)
                    eval_loss += loss.data.item()
                    _, predicted = torch.max(out.data, 1)
                    correct += (predicted == batch_y).sum().item()
                    total += len(batch_y)
                else:
                    break
        loss_val = eval_loss / total * self._args.validate_batch_size
        accuracy = float(correct) / total
        return loss_val, accuracy

    def _update_model_wrapper(self, upload_model):
        if self._trained_model_wrapper.value is not None:
            upload_model.version = self._trained_model_wrapper.value.version + 1
            self._trained_model_wrapper.value = upload_model
        else:
            self._trained_model_wrapper.value = upload_model
        self._score = upload_model.loss

    def _dump_model(self, version):
        torch.save(self._model.state_dict(),
                   '{}/model_params_ps_{}_version_{}_sec_{}.pkl'.format(
                       self._model_dir,
                       self._ps_id,
                       version,
                       timestamps(self._start_time)))


def timestamps(start_time):
    return int(time.time() - start_time)


def bool_parser(v):
    if v.lower() in ('true', '1'):
        return True
    elif v.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
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
    parser.add_argument(
        '--free-trial-steps',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before sync up with the ps')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--validate_batch_size', type=int, default=64,
                        help='batch size for validation dataset in ps')
    parser.add_argument('--validate_max_batch', type=int, default=5,
                        help='max batch for validate model in ps')
    parser.add_argument('--loss-file', default='swamp_metrics_t_{}_pr_{}.png',
                        help='the name of loss figure file')
    parser.add_argument(
        '--loss-sample-interval',
        type=int,
        default=1,
        help='how many batches to wait before record a loss value')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=50,
        metavar='N',
        help='how many batches to wait before logging training status')
    parser.add_argument('--pull-probability', type=float, default=0,
                        help='the probability of trainer pulling from ps')
    parser.add_argument(
        '--crossover-probability',
        type=float,
        default=0,
        help='the probability of crossover op for model pulling')
    parser.add_argument('--trainer-number', type=int, default=1,
                        help='the total number of trainer to launch')
    parser.add_argument('--job-root-dir', default='jobs',
                        help='The root directory of all job result data')
    parser.add_argument(
        '--job-name',
        default=None,
        help='experiment name used for the result data dir name')
    parser.add_argument('--delete-job-data', type=bool_parser, default=True,
                        help='if delete experiment job result data at last.')
    parser.add_argument('--plot-validate-batch-size', type=int, default=64,
                        help='batch size for validation dataset in ps')
    parser.add_argument('--plot-validate-max-batch', type=int, default=5,
                        help='max batch for validate model in ps')
    return parser.parse_args()


def start_ps(
        job_dir,
        args,
        up,
        manager,
        trained_model):
    # Init PS process
    key = 'ps'
    # Shared list used by the parent process and trainer for
    # loss tracing
    ps = PS(job_dir, 0, args, trained_model, up)
    ps_proc = Process(target=ps.run, name='ps')
    ps_proc.start()

    return ps_proc


def start_trainers(
        job_dir,
        args,
        up,
        manager,
        trained_model):
    # Init trainer processes
    trainers = []
    trainer_procs = []
    for t in range(args.trainer_number):
        trainer = Trainer(
            job_dir,
            t,
            args,
            trained_model,
            up)
        trainer_proc = Process(target=trainer.train)
        trainer_proc.start()
        trainers.append(trainer)
        trainer_procs.append(trainer_proc)

    return trainers, trainer_procs


def prepare_validation_loader(batch_size):
    return torch.utils.data.DataLoader(
        datasets.MNIST('./data',
                       train=False,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size,
        shuffle=False)


def validate(data_loader, model, max_batch, batch_size):
    eval_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(data_loader):
            if batch_idx < max_batch:
                out = model(batch_x)
                loss = F.nll_loss(out, batch_y)
                eval_loss += loss.data.item()
                _, predicted = torch.max(out.data, 1)
                correct += (predicted == batch_y).sum().item()
                total += len(batch_y)
            else:
                break
    loss_val = round(eval_loss / total * batch_size, 6)
    accuracy = round(float(correct) / total, 6)
    return loss_val, accuracy


def metric_key_func(metric):
    return metric.timestamp


def recompute_metrics(job_dir, max_validate_batch, validate_batch_size):
    model = torch.load(job_dir + '/model.pkl')
    metrics_dict = {}

    # Prepare data source
    validation_ds = prepare_validation_loader(validate_batch_size)

    # Start recomputing
    start_time = time.time()
    for root, _, files in os.walk(job_dir):
        for f in files:
            if f.startswith('model_params'):
                # Load params and parse meta info.
                model.load_state_dict(torch.load('{}/{}'.format(root, f)))
                meta = f.split('.')[0].split('_')
                model_owner = meta[2] + '_' + meta[3]
                if (meta[2] == 'ps'):
                    msg_info = 'validating ps {} model version {} ...'.format(
                        meta[3], meta[5])
                else:
                    msg_info = 'validating trainer {} epoch {} batch {} ...'.format(
                        meta[3], meta[5], meta[7])
                # Init metircs list
                if model_owner in metrics_dict:
                    metrics = metrics_dict[model_owner]
                else:
                    metrics = []
                    metrics_dict[model_owner] = metrics

                # Compute loss and accuracy.
                print(msg_info)
                loss, accuracy = validate(
                    validation_ds, model, max_validate_batch, validate_batch_size)
                metrics.append(Metrics(loss, accuracy, int(meta[-1])))
    end_time = time.time()
    total_cost = int(end_time - start_time)
    print('recomputing metrics total cost {} seconds'.format(total_cost))

    # Sorting the metrics according timestamps in ascending order.
    for k, v in metrics_dict.items():
        v.sort(key=metric_key_func)

    return metrics_dict


def draw(args, job_dir, metrics_dict):
    image_path = job_dir + '/' + \
        args.loss_file.format(args.trainer_number, args.pull_probability)
    print("Write image to ", image_path)
    lowest_loss, best_accuracy = find_best_metrics_in_ps(metrics_dict)
    fig = plot.figure()

    # Plot the loss/timestamp curve.
    loss_ax = fig.add_subplot(2, 1, 1)
    loss_ax.set_title(
        'swamp training for mnist data (pull probability %s)' %
        args.pull_probability, fontsize=10, verticalalignment='center')
    loss_ax.set_xlabel('timestamp')
    loss_ax.set_ylabel('loss')
    for (k, v) in metrics_dict.items():
        if k.endswith('pull'):
            loss_ax.scatter(timestamps_dict[k], v, s=12, label=k)
        elif k.startswith('ps'):
            losses = [m.loss for m in v]
            timestamps = [m.timestamp for m in v]
            loss_ax.plot(
                timestamps, losses, label=(
                    k + ' (lowest-loss: ' + str(lowest_loss) + ')'))
        else:
            losses = [m.loss for m in v]
            timestamps = [m.timestamp for m in v]
            loss_ax.plot(timestamps, losses, label=k)
    loss_ax.legend(loc='upper right', prop={'size': 6})

    # Plot the accuracy/timestamp curve.
    acc_ax = fig.add_subplot(2, 1, 2)
    acc_ax.set_xlabel('timestamp')
    acc_ax.set_ylabel('accuracy')
    for (k, v) in metrics_dict.items():
        if k.endswith('pull'):
            continue
        elif k.startswith('ps'):
            acc = [m.accuracy for m in v]
            timestamps = [m.timestamp for m in v]
            acc_ax.plot(
                timestamps, acc, label=(
                    k + ' (best-acc: ' + str(best_accuracy) + ')'))
        else:
            acc = [m.accuracy for m in v]
            timestamps = [m.timestamp for m in v]
            acc_ax.plot(timestamps, acc, label=k)
    acc_ax.legend(loc='lower right', prop={'size': 6})

    if args.delete_job_data and os.path.exists(job_dir):
        shutil.rmtree(job_dir)
        os.makedirs(job_dir)

    plot.tight_layout()
    plot.savefig(image_path)


def find_best_metrics_in_ps(metrics_dict):
    loss = float("inf")
    accuracy = 0
    for k, m in metrics_dict.items():
        if k.startswith('ps'):
            # Lookup in this single PS
            min_loss = min(m, key=lambda x: x.loss).loss
            better_acc = max(m, key=lambda x: x.accuracy).accuracy
            if min_loss < loss:
                loss = min_loss
            if better_acc > accuracy:
                accuracy = better_acc
    return loss, accuracy


def main():
    args = parse_args()
    job_name = None
    ps_job_name = None
    if args.job_name is not None:
        job_name = args.job_name
    else:
        job_name = 'swamp_t{}_pr{}'.format(
            args.trainer_number, args.pull_probability)

    job_dir = args.job_root_dir + '/' + job_name
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
    os.makedirs(job_dir)

    torch.manual_seed(args.seed)

    # Persist only one net topology to re-compute metrics before draw curves.
    torch.save(Net(), job_dir + '/model.pkl')

    # Data stores shared by PS, trainers and the main process
    up = Queue()
    manager = Manager()
    trained_model = manager.Value(py_object, None)

    # Start PS and trainers
    ps_proc = start_ps(
        job_dir,
        args,
        up,
        manager,
        trained_model)
    trainers, trainer_procs = start_trainers(
        job_dir, args, up, manager,
        trained_model)

    for proc in trainer_procs:
        proc.join()
    ps_proc.terminate()

    if args.loss_file is not None:
        metrics_dict = recompute_metrics(
            job_dir,
            args.plot_validate_max_batch,
            args.plot_validate_batch_size)
        draw(args, job_dir, metrics_dict)


if __name__ == '__main__':
    main()
