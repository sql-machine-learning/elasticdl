from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import pickle
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager, Value
from ctypes import py_object, c_bool
import queue
import time
import gc
import random
import os
import shutil
from models.network import MNISTNet, CIFAR10Net 
from models.resnet import ResidualBlock, ResNet, resnet18 
from common import prepare_data_loader
from common import ModelLogger
from common import METRICS_IMAGE_FILE_TEMPLATE
from common import JOB_NAME_TEMPLATE
from common import bool_parser


class TrainedModel(object):
    ''' Model uploaded to PS by trainers
    '''

    def __init__(self, model_state, loss=float("inf"), version=1):
        self.model_state = model_state
        self.loss = loss
        self.version = version


class Trainer(object):

    def __init__(
            self,
            job_dir,
            tid,
            args,
            trained_model_wrapper,
            up,
            gpu_id):
        """ Initialize the Trainer.

        Arguments:
          job_dir: Path for storing experimental data.
          tid: The unique identifier of the trainer.
          args: Runtime arguments.
          trained_model_wrapper: The info(eg. state, loss) of the best uploaded
                                 model at present model shared by PS and trainer
                                 which is managed by Manager.
          up: A shared Queue for trainer upload model to PS.
          gpu_id: GPU device id if gpu available.
        """
        self.tid = tid
        self._args = args
        self._up = up
        self._start_time = time.time()
        self._model_class = globals()[args.model_name]
        self._score = float("inf")
        self._trained_model_wrapper = trained_model_wrapper
        self._model_logger = ModelLogger(job_dir)
        self._model_logger.init_trainer_model_dir(tid)
        self._loss_fn = nn.CrossEntropyLoss()
        self._gpu_id = gpu_id
        self._gpu_device = None
        self._last_pull_model_version = -1

    def train(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self._gpu_id)
        self._model = self._model_class()
        self._model.train(True)

        # Must move model into cuda before construct optimizer.
        if self._args.use_gpu and torch.cuda.is_available():
            self._gpu_device = torch.device('cuda:0')
            self._model.to(self._gpu_device)
        self._optimizer = optim.SGD(self._model.parameters(), lr=self._args.lr,
            momentum=self._args.momentum, weight_decay=5e-4)
        self._lr_scheduler = lr_scheduler.MultiStepLR(
            self._optimizer, milestones=[60,90,120], gamma=0.1)

        data_loader = prepare_data_loader(True, self._args.batch_size,
                                          True, self._args.data_type)
        step = 0

        # start local training
        for epoch in range(self._args.epochs):
            self._lr_scheduler.step()
            for batch_idx, (data, target) in enumerate(data_loader):
                if self._args.use_gpu and torch.cuda.is_available():
                    data = data.to(self._gpu_device)
                    target = target.to(self._gpu_device)
                self._optimizer.zero_grad()
                output = self._model(data)
                loss = self._loss_fn(output, target)
                
                if step < self._args.free_trial_steps:
                    loss.backward()
                    self._optimizer.step()
                    step = step + 1
                else:
                    if loss.data.item() < self._score:
                        self._push_model(loss)
                    else:
                        if random.random() < self._args.pull_probability:
                            self._pull_model()
                    step = 0

                gc.collect()
                if batch_idx % self._args.model_sample_interval == 0:
                    self._model_logger.dump_model_in_trainer(
                        self._model.state_dict(), self.tid, epoch, batch_idx)
                self._print_progress(epoch, batch_idx)

            # Push model at the end of each epoch.
            self._push_model(loss)
            print("trainer %i done epoch %i" % (self.tid, epoch))

    def _pull_model(self):
        trained_model = self._trained_model_wrapper.value

        if self._last_pull_model_version >= trained_model.version:
            return

        self._last_pull_model_version = trained_model.version
        coefficient = random.random()
        s_dict = self._model.state_dict()
        for k in trained_model.model_state:
            if self._args.use_gpu and torch.cuda.is_available():
                trained_value = trained_model.model_state[k].to(self._gpu_device)
            else:
                trained_value = trained_model.model_state[k]
            s_dict[k] = s_dict[k] * coefficient + trained_value * (1 - coefficient)
        self._score = trained_model.loss

    def _push_model(self, loss):
        state_dict_copy = {}
        for k, v in self._model.state_dict().items():
           state_dict_copy[k] = v.to('cpu')
        self._score = loss.data.item()
        if self._up is not None:
            upload_model = TrainedModel(
                state_dict_copy, loss.data.item())
            self._up.put(pickle.dumps(upload_model))

    def _print_progress(self, epoch, batch_idx):
        if batch_idx % self._args.log_interval == 0:
            print("Current trainer id: %i, epoch: %i, batch id: %i" %
                  (self.tid, epoch, batch_idx))


class PS(object):

    def __init__(
            self,
            job_dir,
            args,
            trained_model_wrapper,
            up,
            stop_ps,
            gpu_id):
        """ Initialize the PS.

        Arguments:
          job_dir: Path for storing experimental data.
          args: Runtime arguments.
          trained_model_wrapper: The info(eg. state, loss) of the best uploaded
                                 model at present shared by PS and trainer which
                                 is managed by Manager.
          up: A shared Queue for trainer upload model to PS.
          stop_ps: A shared bool value for the main process to stop PS.
          gpu_id: GPU device id if gpu available.
        """
        self._args = args
        self._up = up
        self._start_time = time.time()
        self._model_class = globals()[args.model_name]
        self._trained_model_wrapper = trained_model_wrapper
        self._score = float("inf")
        self._validate_score = float("inf")
        self._model_logger = ModelLogger(job_dir)
        self._model_logger.init_ps_model_dir()
        self._stop_ps = stop_ps
        self._loss_fn = nn.CrossEntropyLoss()
        self._gpu_id = gpu_id
        self._gpu_device = None

    def run(self):
        self._model = self._model_class()
        if self._args.use_gpu and torch.cuda.is_available():
            self._gpu_device = torch.device('cuda:{}'.format(self._gpu_id))
            self._model.to(self._gpu_device)
        updates = 0
        validate_loader = prepare_data_loader(
            True, self._args.batch_size,
            True, self._args.data_type)

        while not self._up.empty() or not self._stop_ps.value:
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
                double_check_loss, accuracy = self._validate(validate_loader, self._gpu_device)
                if double_check_loss < self._validate_score:
                    self._update_model_wrapper(upload_model)
                    self._validate_score = double_check_loss
                    self._model_logger.dump_model_in_ps(
                        self._model.state_dict(), upload_model.version)

    def _validate(self, data_loader, gpu_device):
        max_batch = self._args.validate_max_batch
        eval_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            self._model.train(False)
            for batch_idx, (batch_x, batch_y) in enumerate(data_loader):
                if batch_idx < max_batch:
                    if self._args.use_gpu and torch.cuda.is_available():
                        batch_x = batch_x.to(gpu_device)
                        batch_y = batch_y.to(gpu_device)
                    out = self._model(batch_x)
                    loss = self._loss_fn(out, batch_y)
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


def _parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Swamp Example')
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
    parser.add_argument('--data-type', default='mnist',
                        help='the name of the dataset (mnist, cifar10)')
    parser.add_argument('--model-name', default='MNISTNet',
                        help='the name of the model (MNISTNet, CIFAR10Net, resnet18)')
    parser.add_argument('--loss-file', default=METRICS_IMAGE_FILE_TEMPLATE,
                        help='the name of loss figure file')
    parser.add_argument(
        '--model-sample-interval',
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
    parser.add_argument('--trainer-number', type=int, default=1,
                        help='the total number of trainer to launch')
    parser.add_argument('--job-root-dir', default='jobs',
                        help='The root directory of all job result data')
    parser.add_argument(
        '--job-name',
        default=None,
        help='experiment name used for the result data dir name')
    parser.add_argument('--use-gpu', type=bool_parser, default=True,                                                                
                        help='use GPU for training if available')

    return parser.parse_args()


def _start_ps(
        job_dir,
        args,
        up,
        manager,
        trained_model,
        stop_ps,
        gpu_id):
    # Init PS process
    key = 'ps'
    # Shared list used by the parent process and trainer for
    # loss tracing
    ps = PS(job_dir, args, trained_model, up, stop_ps, gpu_id)
    ps_proc = Process(target=ps.run, name='ps')
    ps_proc.start()

    return ps_proc


def _start_trainers(
        job_dir,
        args,
        up,
        manager,
        trained_model,
        total_gpu_cnt):
    # Init trainer processes
    trainers = []
    trainer_procs = []
    for t in range(args.trainer_number):
        trainer = Trainer(
            job_dir,
            t,
            args,
            trained_model,
            up,
            0 if total_gpu_cnt == 0 else t % total_gpu_cnt)
        trainer_proc = Process(target=trainer.train)
        trainer_proc.start()
        trainers.append(trainer)
        trainer_procs.append(trainer_proc)

    return trainers, trainer_procs


def _prepare():
    args = _parse_args()
    mp.set_start_method('spawn', force=True)
    torch.manual_seed(args.seed)
    job_name = None
    if args.job_name is not None:
        job_name = args.job_name
    else:
        job_name = JOB_NAME_TEMPLATE.format(
            args.trainer_number, args.pull_probability)

    job_dir = args.job_root_dir + '/' + job_name
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
    os.makedirs(job_dir)

    with open(job_dir + '/meta.info', 'w') as meta:
        meta.write('{}_{}'.format(args.trainer_number, args.pull_probability))

    return args, job_dir


def _train(args, job_dir):
    # Data stores shared by PS, trainers and the main process
    up = Queue()
    manager = Manager()
    trained_model = manager.Value(py_object, None)
    stop_ps = Value(c_bool, False)

    # Save model net.
    model_class = globals()[args.model_name]
    torch.save(model_class(), job_dir + '/model.pkl')

    # Available GPU count.
    total_gpu_cnt = torch.cuda.device_count() 

    # Start PS and trainers.
    ps_proc = _start_ps(
        job_dir,
        args,
        up,
        manager,
        trained_model,
        stop_ps,
        0)
    trainers, trainer_procs = _start_trainers(
        job_dir, args, up, manager,
        trained_model, total_gpu_cnt)

    for proc in trainer_procs:
        proc.join()

    stop_ps.value = True
    ps_proc.join()


def main():
    args, job_dir = _prepare()
    _train(args, job_dir)


if __name__ == '__main__':
    main()
