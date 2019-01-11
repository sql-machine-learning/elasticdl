import os
import time
import torch
from torchvision import datasets, transforms

TRAINER_MODEL_FILE_TEMPLATE = '{}/model_params_trainer_{}_epoch_{}_batch_{}_sec_{}.pkl'
TRAINER_MODEL_DIR = '{}/trainer_{}'

PS_MODEL_FILE_TEMPLATE = '{}/model_params_ps_model_version_{}_sec_{}.pkl'
PS_MODEL_DIR = '{}/ps'

METRICS_IMAGE_FILE_TEMPLATE = 'swamp_metrics_t_{}_pp_{}.png'
JOB_NAME_TEMPLATE = 'swamp_t{}_pp{}'


def prepare_data_loader(is_train, batch_size, shuffle, data='mnist', data_dir='./data'):
    if data == 'mnist':
        dataset = datasets.MNIST(data_dir,
                       train=is_train,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    elif data == 'cifar10':
        if is_train:
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        else:
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        dataset = datasets.CIFAR10(data_dir,
                       train=is_train,
                       download=True,
                       transform=transform
                       )
    elif data == 'ImageNet':
        if is_train:
            transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        dataset = datasets.ImageFolder(data_dir, transform)
    else:
        raise AttributeError('data %s not supported' % data)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle)


def bool_parser(v):
    if v.lower() in ('true', '1'):
        return True
    elif v.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def _time_diff(start_time):
    return int(time.time() - start_time)


class ModelLogger(object):

    def __init__(self, job_dir):
        self._job_dir = job_dir
        self._start_time = time.time()

    def init_trainer_model_dir(self, trainer_id):
        model_dir = TRAINER_MODEL_DIR.format(self._job_dir, trainer_id)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def init_ps_model_dir(self):
        model_dir = PS_MODEL_DIR.format(self._job_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def dump_model_in_trainer(self, model_state, trainer_id, epoch, batch_idx):
        torch.save(
            model_state,
            TRAINER_MODEL_FILE_TEMPLATE.format(
                TRAINER_MODEL_DIR.format(self._job_dir, trainer_id),
                trainer_id,
                epoch,
                batch_idx,
                _time_diff(
                    self._start_time)))

    def dump_model_in_ps(self, model_state, model_version):
        torch.save(
            model_state,
            PS_MODEL_FILE_TEMPLATE.format(
                PS_MODEL_DIR.format(self._job_dir, model_version),
                model_version,
                _time_diff(
                    self._start_time)))
