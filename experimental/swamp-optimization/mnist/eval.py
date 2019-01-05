from __future__ import print_function
import os
import sys
import argparse
import torch
from torchvision import datasets, transforms
import time
import os
import shutil
import network
import torch.nn.functional as F
import multiprocessing
from multiprocessing import Process, Queue
import queue
from common import prepare_data_loader
from common import bool_parser


def _validate(data_loader, model, max_batch, batch_size):
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


def _evaluate(job_root_dir, max_validate_batch, validate_batch_size, concurrency, data_type):
    # Prepare data source
    validation_ds = prepare_data_loader(False, validate_batch_size,
                                        False, data_type)
    validation_jobs = Queue()
    net_class = data_type.upper() + 'Net'

    # Evaluate all the jobs under job_root_dir.
    for parent, dirs, _ in os.walk(job_root_dir):
        for job_name in dirs:
            if job_name.startswith('swamp_'):
                job_dir = parent + '/' + job_name

                # Start recomputing
                start_time = time.time()
                for root, _, files in os.walk(job_dir):
                    for f in files:
                        if f.startswith('model_params') and f.endswith('.pkl'):
                            meta = f.split('.')[0].split('_')
                            model_owner = meta[2] + '_' + meta[3]
                            if (meta[2] == 'ps'):
                                msg_info = 'validating job {} ps model version {} ...'.format(
                                    job_name, meta[5])
                            else:
                                msg_info = 'validating job {} trainer {} epoch {} batch {} ...'.format(
                                    job_name, meta[3], meta[5], meta[7])
                            work_params = {
                                'validation_ds': validation_ds,
                                'max_batch': max_validate_batch,
                                'batch_size': validate_batch_size,
                                'job_dir': job_dir,
                                'pkl_dir': root,
                                'param_file': f,
                                'timestamp': meta[-1],
                                'msg': msg_info
                            }
                            validation_jobs.put(work_params)
    # Start validation
    start_time = time.time()
    job_procs = []
    for _ in range(concurrency):
        # Add sentinel job for each evaluation process.
        validation_jobs.put({})
        job = _SingleValidationJob(validation_jobs, net_class) 
        job_proc = Process(target=job.validate)
        job_proc.start()
        job_procs.append(job_proc)

    for proc in job_procs:
        proc.join()

    end_time = time.time()
    total_cost = int(end_time - start_time)
    print('validation metrics total cost {} seconds'.format(total_cost))


class _SingleValidationJob(object):
    def __init__(self, job_queue, net_class):
        self._model = getattr(network, net_class)()
        self._job_queue = job_queue 

    def validate(self):
        while True:
            param_dict = self._job_queue.get()
            # Found sentinel job and eval process could exit.
            if not param_dict:
                break

            print(param_dict['msg'])
            #model = torch.load(param_dict['job_dir'] + '/model.pkl')
            self._model.load_state_dict(torch.load(
                '{}/{}'.format(param_dict['pkl_dir'], param_dict['param_file'])))

            loss, accuracy = _validate(
                param_dict['validation_ds'], self._model, 
                    param_dict['max_batch'], param_dict['batch_size'])
            eval_filename = param_dict['pkl_dir'] + '/' + \
                param_dict['param_file'].split('.')[0] + '.eval'

            if os.path.exists(eval_filename):
                os.remove(eval_filename)
            with open(eval_filename, 'w') as eval_f:
                eval_f.write(
                    '{}_{}_{}'.format(
                        loss, accuracy, int(
                            param_dict['timestamp'])))

def _prepare():
    args = _parse_args()
    torch.manual_seed(args.seed)
    os.putenv('OMP_NUM_THREADS', '1')
    return args


def _parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--job-root-dir', default='jobs',
                        help='The root directory of all job result data')
    parser.add_argument(
        '--job-name',
        default=None,
        help='experiment name used for the result data dir name')
    parser.add_argument('--delete-job-data', type=bool_parser, default=False,
                        help='if delete experiment job result data at last.')
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=64,
        help='batch size for evaluate model logged by train.py')
    parser.add_argument('--data-type', default='mnist',
                        help='the name of the dataset (mnist, cifar10)')
    parser.add_argument('--eval-max-batch', type=int, default=sys.maxsize,
                        help='max batch for evaluate model logged by train.py')
    parser.add_argument('--eval-concurrency', type=int, 
                        default=int(multiprocessing.cpu_count()/2),
                        help='process concurrency for evaluation')
    return parser.parse_args()


def main():
    args = _prepare()
    _evaluate(
        args.job_root_dir,
        args.eval_batch_size,
        args.eval_max_batch,
        args.eval_concurrency,
        args.data_type)


if __name__ == '__main__':
    main()
