import queue
import time
import threading
import numpy as np
from matplotlib import pyplot as plot


class SwampParameterServer(object):
    def __init__(self):
        self._report_count = 0
        self._pull_count = 0
        self._push_count = 0
        self._lock = threading.Lock()
        self._exiting = False
        self._accuracy = 0
        self._vars = {}
        self._log = []

    def start(self):
        self._start_time = time.time()
        self._log.append((0, 0))

    def stop(self):
        self._log.append((time.time() - self._start_time, self._accuracy))

    def accuracy(self):
        return self._accuracy

    def pull(self, names=None):
        if self._accuracy > 0:
            with self._lock:
                if names:
                    res = {k: self._vars[k] for k in names}
                else:
                    res = self._vars.copy()
        else:
            res = {}
        self._pull_count += 1
        return self._accuracy, res

    def push(self, accuracy, vars):
        with self._lock:
            if accuracy > self._accuracy:
                for name in vars:
                    self._vars[name] = vars[name]
                    self._accuracy = accuracy
                self._log.append((time.time() - self._start_time, accuracy))
                print("push %d accuracy %1.6g" % (self._push_count, accuracy))
        self._push_count += 1

    # return wether the reported accuracy is better than current one.
    def report_accuracy(self, accuracy):
        self._report_count += 1
        return accuracy > self._accuracy

    def plot_accuracy_log(self, file_name, info=""):
        timesteps = [l[0] for l in self._log]
        accs = [l[1] for l in self._log]
        plot.plot(timesteps, accs)
        plot.xlabel("Timestep")
        plot.ylabel("Accuracy")
        plot.title("swamp training %s acc=%1.5g" % (info, self._accuracy))
        plot.savefig(file_name)
