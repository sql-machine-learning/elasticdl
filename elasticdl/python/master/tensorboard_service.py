import datetime
import subprocess

import numpy as np
import tensorflow as tf


class TensorboardService(object):
    """Tensorboard Service implementation"""

    def __init__(self, tensorboard_log_dir, master_ip):
        """
        Arguments:
            tensorboard_log_dir: The log directory for Tensorboard.
        """
        _current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self._tensorboard_log_dir = tensorboard_log_dir + _current_time
        self._initialize_summary_writer()
        self._master_ip = master_ip
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
                elif isinstance(v, dict) and v:
                    v = list(v.values())[0]
                tf.summary.scalar(k, v, step=version)

    def start(self):
        # TODO: Find a good way to catch the exception if any.
        # `tb_process.poll()` is unreliable as TensorBoard won't
        # exit immediately in some cases, e.g. when host is missing.
        self.tb_process = subprocess.Popen(
            [
                "tensorboard --logdir %s --host %s"
                % (self._tensorboard_log_dir, self._master_ip)
            ],
            shell=True,
            stdout=subprocess.DEVNULL,
        )

    def is_active(self):
        return self.tb_process.poll() is None
