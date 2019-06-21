import datetime
import subprocess
import time

import tensorflow as tf

import numpy as np


class TensorboardService(object):
    """Tensorboard Service implementation"""

    def __init__(self, tensorboard_log_dir):
        """
        Arguments:
            tensorboard_log_dir: The log directory for Tensorboard.
        """
        _current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self._tensorboard_log_dir = tensorboard_log_dir + _current_time
        self._initialize_summary_writer()
        self.tb_process = None

    def _initialize_summary_writer(self):
        self.summary_writer = tf.summary.create_file_writer(
            self._tensorboard_log_dir
        )

    def write_dict_to_summary(self, dictionary, version):
        with self.summary_writer.as_default():
            for k, v in dictionary.items():
                if isinstance(v, np.ndarray) and len(v) == 1:
                    v = v[0]
                tf.summary.scalar(k, v, step=version)

    def start(self):
        self.tb_process = subprocess.Popen(
            ["tensorboard --logdir " + self._tensorboard_log_dir], shell=True
        )

    def keep_running(self):
        while self.tb_process.poll() is None:
            time.sleep(10)
