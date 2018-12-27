from __future__ import print_function
import argparse
import torch
from torchvision import datasets, transforms
import time
import os
import shutil
from network import Net
import torch.nn.functional as F
import multiprocessing
from multiprocessing import Pool


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
    parser.add_argument('--plot-validate-batch-size', type=int, default=64,
                        help='batch size for validation dataset in ps')
    parser.add_argument('--plot-validate-max-batch', type=int, default=5,
                        help='max batch for validate model in ps')
    return parser.parse_args()


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


def evaluate(job_root_dir, max_validate_batch, validate_batch_size):
    # Prepare data source
    validation_ds = prepare_validation_loader(validate_batch_size)
 
    validation_works = []

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
                                msg_info = 'validating job {} ps {} model version {} ...'.format(
                                    job_name, meta[3], meta[5])
                            else:
                                msg_info = 'validating job {} trainer {} epoch {} batch {} ...'.format(
                                    job_name, meta[3], meta[5], meta[7])
                            work_params = {
                                'validation_ds' : validation_ds,
                                'max_batch' : max_validate_batch,
                                'batch_size' : validate_batch_size,
                                'job_dir' : job_dir,
                                'pkl_dir' : root,
                                'param_file' : f,
                                'timestamp' : meta[-1],
                                'msg' : msg_info
                            }
                            validation_works.append(work_params)
    # Start validation 
    start_time = time.time()
    pool = Pool(processes=int(multiprocessing.cpu_count()/2))
    pool.map(single_validate, validation_works)
    pool.close()
    pool.join()
    end_time = time.time()
    total_cost = int(end_time - start_time)
    print('validation metrics total cost {} seconds'.format(total_cost))

def single_validate(param_dict):
    print(param_dict['msg'])
    model = torch.load(param_dict['job_dir'] + '/model.pkl')
    model.load_state_dict(torch.load('{}/{}'.format(param_dict['pkl_dir'], param_dict['param_file'])))    
    loss, accuracy = validate(
        param_dict['validation_ds'], model, param_dict['max_batch'], param_dict['batch_size'])
    eval_filename = param_dict['pkl_dir'] + '/' + param_dict['param_file'].split('.')[0] + '.eval'
    if os.path.exists(eval_filename):
        os.remove(eval_filename)
    with open(eval_filename, 'w') as eval_f:
        eval_f.write('{}_{}_{}'.format(loss, accuracy, int(param_dict['timestamp']))) 


def prepare():
    args = parse_args()
    torch.manual_seed(args.seed)
    return args


def main():
    args = prepare()
    evaluate(
        args.job_root_dir,
        args.plot_validate_max_batch,
        args.plot_validate_batch_size)


if __name__ == '__main__':
    main()
