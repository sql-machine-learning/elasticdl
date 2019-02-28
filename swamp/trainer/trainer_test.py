import unittest
import threading 
import time
import os
from unittest import mock
from trainer import Task, Trainer

class TrainerTest(unittest.TestCase):

    def testTrainModel(self):
        
        training_params = {
            'use_gpu' : True,
            'log_interval' : 50,
            'epochs' : 1}
        hyper_params = {
            'batch_size' : 64,
            'lr' : 0.1,
            'momentum' : 0.5,
            'weight_decay' : 5e-4,
            'lr_sched_milestones' : [10, 20, 30],
            'lr_sched_gamma' : 0.1}
        task = Task('dev_job',
            'task_1',
            'MNISTNet',
            'jobs/dev_job/task_1/network.py',
            'jobs/dev_job/task_1/model.pkl',
            'loss_func',
            'optimizer_func',
            'jobs/dev_job/task_1/data.tar.gz',
            None,
            'preprocess_data',
            None,
            'prepare_training_dataset',
            'prepare_validation_dataset',
            training_params,
            hyper_params)

        if os.path.exists('model.pkl'):
            os.remove('model.pkl')
        trainer = Trainer('AT-20ANT') 
        trainer._do_fetch = mock.Mock(return_value=task)
        trainer._upload_file = mock.Mock(return_value=None)
        threading.Thread(target=trainer.start).start()
        time.sleep(5)
        threading.Thread(target=self._stop_trainer, args=(trainer,)).start()
        while True:
            if os.path.exists('model.pkl'):
                break
        os.remove('model.pkl')

    def _stop_trainer(self, trainer):
        trainer.stop()

if __name__ == '__main__':
    unittest.main()
