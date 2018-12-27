from __future__ import print_function
import argparse
import sys
from matplotlib import pyplot
import os
import shutil


class Metrics(object):
    def __init__(self, loss, accuracy, timestamp=1):
        self.loss = loss
        self.accuracy = accuracy
        self.timestamp = timestamp


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--loss-file', default='swamp_metrics_t_{}_pr_{}.png',
                        help='the name of loss figure file')
    parser.add_argument('--job-root-dir', default='jobs',
                        help='The root directory of all job result data')
    return parser.parse_args()


def metric_key_func(metric):
    return metric.timestamp


def sort_image_file(filename):
    return filename.split('/')[-1].split('_')[3]

def parse_job_meta(job_dir):
    with open(job_dir + '/meta.info', 'r') as meta:
        meta_infos = meta.readline().split('_')        
        return meta_infos[0], meta_infos[1]

def plot(args, job_root_dir, all_job_metrics_dict):
    for job_name, metrics_dict in all_job_metrics_dict.items():
        job_dir = job_root_dir + '/' + job_name
        trainer_number, pull_probability = parse_job_meta(job_dir) 
        image_path = job_dir + '/' + \
            args.loss_file.format(trainer_number, pull_probability)
        print("Write image to ", image_path)
        lowest_loss, best_accuracy = find_best_metrics_in_ps(metrics_dict)
        fig = pyplot.figure()

        # Plot the loss/timestamp curve.
        loss_ax = fig.add_subplot(2, 1, 1)
        loss_ax.set_title(
            'swamp training for mnist data (pull probability %s)' %
            pull_probability, fontsize=10, verticalalignment='center')
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

        pyplot.tight_layout()
        pyplot.savefig(image_path)


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


def collect_metrics(job_root_dir): 
    # Collect all the jobs's evaluation data.
    all_job_metrics_dict = {}
    for parent, dirs, _ in os.walk(job_root_dir):
        for job_name in dirs:
            if job_name.startswith('swamp_'):
                job_dir = parent + '/' + job_name
                metrics_dict = {}
                all_job_metrics_dict[job_name] = metrics_dict

                # Start to collect. 
                for root, _, files in os.walk(job_dir):
                    for f in files:
                        if f.startswith('model_params') and f.endswith('.eval'): 
                            metrics_owner = root.split('/')[-1]
                            if metrics_owner not in metrics_dict:
                                metrics = []
                                metrics_dict[metrics_owner] = metrics
                            with open(root + '/' + f, 'r') as eval_f:
                                metrics_vals = eval_f.readline().split('_')
                                metrics_dict[metrics_owner].append(Metrics(float(metrics_vals[0]), float(metrics_vals[1]), int(metrics_vals[2])))

                # Sorting the metrics according timestamps in ascending order.
                for k, v in metrics_dict.items():
                    v.sort(key=metric_key_func)

    return all_job_metrics_dict

def main():
    args = parse_args()
    all_job_metrics_dict = collect_metrics(args.job_root_dir)
    plot(args, args.job_root_dir, all_job_metrics_dict)


if __name__ == '__main__':
    main()
