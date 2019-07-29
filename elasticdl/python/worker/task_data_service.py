import logging
import threading
from contextlib import closing

import recordio
import tensorflow as tf

from elasticdl.proto import elasticdl_pb2


class TaskDataService(object):
    def __init__(self, worker):
        self._logger = logging.getLogger(__name__)
        self._worker = worker
        self._record_count = 0
        self._reported_record_count = 0
        self._pending_tasks = []
        self._lock = threading.Lock()
        self._current_task = None
        self._pending_dataset = True

    def get_current_task(self):
        return self._current_task

    def report_record_done(self, count, err_msg=""):
        self._reported_record_count += count
        if (
            len(self._pending_tasks)
            and self._reported_record_count >= self._pending_tasks[0][1]
        ):
            with self._lock:
                while (
                    len(self._pending_tasks)
                    and self._reported_record_count
                    >= self._pending_tasks[0][1]
                ):
                    task, _ = self._pending_tasks.pop(0)
                    self._worker.report_task_result(task.task_id, err_msg)
                if len(self._pending_tasks):
                    self._current_task = self._pending_tasks[0][0]

    def get_dataset(self):
        if self._pending_dataset:
            ds = tf.data.Dataset.from_generator(
                self.gen, (tf.string), (tf.TensorShape([]))
            )
            self._pending_dataset = False
            return ds
        else:
            return None

    def gen(self):
        while True:
            task = self._worker.get_task()
            if not task.shard_file_name:
                if task.type == elasticdl_pb2.WAIT:
                    self._pending_dataset = True
                    self._logger.info(
                        "Finish current dataset, maybe more data later"
                    )
                else:
                    self._logger.info("No more task, stopping")
                break
            with self._lock:
                self._record_count += task.end - task.start
                self._pending_tasks.append((task, self._record_count))
                if len(self._pending_tasks) == 1:
                    self._current_task = task
            with closing(
                recordio.Scanner(
                    task.shard_file_name, task.start, task.end - task.start
                )
            ) as reader:
                while True:
                    r = reader.record()
                    if r:
                        yield r
                    else:
                        break
