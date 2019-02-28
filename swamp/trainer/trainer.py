import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from queue import Queue
import importlib
import threading
import subprocess


class Task(object):

    def __init__(
            self,
            job_id,
            task_id,
            model_name,
            model_path,
            trained_model_path,
            loss_func,
            optimizer_func,
            training_data_path,
            validation_data_path,
            training_data_preprocess_func,
            validation_data_preprocess_func,
            training_dataset_func,
            validation_dataset_func,
            training_params,
            hyper_params):
        self.job_id = job_id
        self.task_id = task_id
        self.model_name = model_name
        self.model_path = model_path
        self.trained_model_path = trained_model_path
        self.validation_data_path = validation_data_path
        self.loss_func = loss_func
        self.optimizer_func = optimizer_func
        self.training_data_path = training_data_path
        self.training_data_preprocess_func = training_data_preprocess_func
        self.validation_data_preprocess_func = validation_data_preprocess_func
        self.training_dataset_func = training_dataset_func
        self.validation_dataset_func = validation_dataset_func
        self.training_params = training_params
        self.hyper_params = hyper_params

class TaskResult(object):

    def __init__(
            self,
            model_path,
            loss,
            accuracy):
        self.model_path = model_path
        self.loss = loss
        self.accuracy = accuracy

class Trainer(object):

    def __init__(
            self,
            pangu_cluster):
        """ Initialize the Trainer.
        """
        self._pangu_cluster = pangu_cluster 
        self._task_queue = Queue()
        self._result_queue = Queue()
        self._stop = False

    def start(self):
        # start task fetcher subprocess.
        self._start_task_fetcher()

        # process tasks in pending task queue.
        while not self._stop:
            task = self._task_queue.get()

            # download model
            self._download_file(task.model_path)

            # import user-defined module dynamically
            module_path_segs = task.model_path.split('/')
            local_model_name = module_path_segs[len(module_path_segs) - 1]
            module = importlib.import_module(local_model_name.split('.')[0])

            model = self._train(module, task)
            loss, acc = self._evaluate(module, model, task)
            print("job %s task %s loss %f accuracy %f" % 
                    (task.job_id, task.task_id, loss, acc))
            self._save_model(model, task.trained_model_path)
            result = TaskResult(task.trained_model_path, loss, acc)
            self._result_queue.put(result)

    def stop(self):
        self._stop = True

    def _start_task_fetcher(self):
         task_fetcher = threading.Thread(target=self._fetch_task)
         task_fetcher.start()

    def _train(self, module, task):
        # downlaod training data
        self._download_file(task.training_data_path)

        model_class = getattr(module, task.model_name)
        model = model_class()
        model.train(True)

        # preprocess training data
        getattr(module, task.training_data_preprocess_func)()

        # get user-defined training parameters
        use_gpu = task.training_params['use_gpu']
        epochs = task.training_params['epochs']
        log_interval = task.training_params['log_interval']

        # get user-defined hyper-parameters
        batch_size = task.hyper_params['batch_size']
        lr = task.hyper_params['lr']
        momentum = task.hyper_params['momentum']
        weight_decay = task.hyper_params['weight_decay']
        milestones = task.hyper_params['lr_sched_milestones']
        gamma = task.hyper_params['lr_sched_gamma']

        # init loss func and optimizer
        loss_fn = getattr(module, task.loss_func)()
        optimizer = getattr(module, task.optimizer_func)(
                model.parameters(), lr,
                momentum, weight_decay)

        # Must move model into cuda before construct optimizer.
        if use_gpu and torch.cuda.is_available():
            self._gpu_device = torch.device('cuda:0')
            model.to(self._gpu_device)

        lr_sched = lr_scheduler.MultiStepLR(
            optimizer, milestones, gamma)

        # init dataloader
        dataset = getattr(module, task.training_dataset_func)()
        data_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

        # start local training
        for epoch in range(epochs):
            lr_sched.step()
            for batch_idx, (data, target) in enumerate(data_loader):
                if use_gpu and torch.cuda.is_available():
                    data = data.to(self._gpu_device)
                    target = target.to(self._gpu_device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                self._print_progress(task.job_id, task.task_id, epoch, batch_idx, log_interval)

            print("job %s task %s done epoch %i" % (task.job_id, task.task_id, epoch))
        return model

    def _evaluate(self, module, model, task):
        # download validation data if any.
        if task.validation_data_path and len(task.validation_data_path.strip()) > 0:
            self._download_file(self._pangu_cluster, task.validation_data_path)
        if task.validation_data_preprocess_func:
            getattr(module, task.validation_data_preprocess_func)()

        use_gpu = task.training_params['use_gpu']
        batch_size = task.hyper_params['batch_size']
        loss_fn = getattr(module, task.loss_func)()

        # init validation dataset
        dataset = getattr(module, task.validation_dataset_func)()
        data_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
        if use_gpu:
            device = torch.device('cuda:0')

        # start evaluating.
        eval_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(data_loader):
                if use_gpu:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                out = model(batch_x)
                loss = loss_fn(out, batch_y)
                eval_loss += loss.data.item()
                _, predicted = torch.max(out.data, 1)
                correct += (predicted == batch_y).sum().item()
                total += len(batch_y)
        loss_val = round(eval_loss / total * batch_size, 6)
        accuracy = round(float(correct) / total, 6)
        return loss_val, accuracy

    def _save_model(self, model, remote_file_path):
        torch.save(model, 'model.pkl')
        self._upload_file('model.pkl', remote_file_path)

    def _print_progress(self, job_id, task_id, epoch, batch_idx, log_interval):
        if batch_idx % log_interval == 0:
            print("job %s task %s epoch: %i, batch id: %i" %
                  (job_id, task_id, epoch, batch_idx))

    def _fetch_task(self):
        task = self._do_fetch()
        self._task_queue.put(task) 

    def _do_fetch(self):
        # TODO: use protobuf client to fetch task from coordinator
        pass

    def _download_file(self, remote_file_path):
        path_split_segs = remote_file_path.split('/')
        local_file = path_split_segs[len(path_split_segs) - 1] 
        cmd = 'python /pangu/pangu_cli.py cp pangu://' + \
            self._pangu_cluster + '/elasticdl/' + remote_file_path + ' ' + local_file
        print('executing download cmd: ' + cmd)
        status, result = subprocess.getstatusoutput(cmd)
        print('download model status: ' + str(status))
        print('download model output: ' + result)

    def _upload_file(self, local_file_path, remote_file_path):
        cmd = 'python /pangu/pangu_cli.py cp ' + local_file_path + ' pangu://' + \
            self._pangu_cluster + '/elasticdl/' + remote_file_path
        print('executing upload cmd: ' + cmd)
        status, result = subprocess.getstatusoutput(cmd)
        print('upload model status: ' + str(status))
        print('upload model output: ' + result)

def _parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Swamp Example')
    parser.add_argument('--pangu-cluster',
                        help='pangu cluster name for model and data storage')
    return parser.parse_args()

def _prepare():
    return _parse_args()

def main():
    args = _prepare()
    trainer = Trainer(args.pangu_cluster)
    trainer.start()

if __name__ == '__main__':
    main()
