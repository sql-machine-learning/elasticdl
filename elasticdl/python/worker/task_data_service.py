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
        self._pending_tasks_with_data_counts = []
        self._lock = threading.Lock()
        self._current_task = None
        self._pending_dataset = True

    def get_current_task(self):
        return self._current_task

    def report_record_done(self, count, err_msg=""):
        """
        Report the number of records in the latest processed batch,
        so TaskDataService knows if some pending tasks are finished
        and report_task_result to the master.
        self._pending_tasks_with_data_counts[0][0] is the first pending task,
        self._pending_tasks_with_data_counts[0][1] is the number of records
        in this task.
        """
        self._reported_record_count += count
        if (
            len(self._pending_tasks_with_data_counts)
            and self._reported_record_count
            >= self._pending_tasks_with_data_counts[0][1]
        ):
            with self._lock:
                while (
                    len(self._pending_tasks_with_data_counts)
                    and self._reported_record_count
                    >= self._pending_tasks_with_data_counts[0][1]
                ):
                    task, _ = self._pending_tasks_with_data_counts.pop(0)
                    self._worker.report_task_result(task.task_id, err_msg)
                if len(self._pending_tasks_with_data_counts):
                    self._current_task = self._pending_tasks_with_data_counts[
                        0
                    ][0]

    def get_dataset(self):
        """
        Return a RecordIO dataset, or None if no more data.
        """
        if self._pending_dataset:
            ds = tf.data.Dataset.from_generator(
                self.gen, (tf.string), (tf.TensorShape([]))
            )
            self._pending_dataset = False
            return ds
        else:
            return None

    def gen(self):
        """
        A generator supports the iter() protocol (e.g. a generator function),
        which is used to create a dataset for RecordIO.
        """
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
                self._pending_tasks_with_data_counts.append(
                    (task, self._record_count)
                )
                if len(self._pending_tasks_with_data_counts) == 1:
                    self._current_task = task
            with closing(
                recordio.Scanner(
                    task.shard_file_name, task.start, task.end - task.start
                )
            ) as reader:
                while True:
                    record = reader.record()
                    if record:
                        yield record
                    else:
                        break
