import queue
import threading
from collections import deque

import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.constants import TaskExecCounterKey
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.data.reader.data_reader_factory import create_data_reader


class TaskDataService(object):
    def __init__(
        self, worker, training_with_evaluation, data_reader_params=None
    ):
        self._worker = worker
        self._create_data_reader_fn = create_data_reader
        if self._worker._custom_data_reader is not None:
            self._create_data_reader_fn = self._worker._custom_data_reader
        self._training_with_evaluation = training_with_evaluation
        self._lock = threading.Lock()
        self._pending_dataset = True
        self._pending_save_model_task = None
        if data_reader_params:
            self.data_reader = self._create_data_reader_fn(
                data_origin=None, **data_reader_params
            )
        else:
            self.data_reader = self._create_data_reader_fn(data_origin=None)
        self._warm_up_task = None
        self._has_warmed_up = False
        self._failed_record_count = 0
        self._reported_record_count = 0
        self._current_task = None
        self._pending_tasks = deque()
        self._records_queue = queue.Queue(maxsize=2)

    def _reset(self):
        """
        Reset pending tasks and record counts
        """
        self._reported_record_count = 0
        self._failed_record_count = 0
        self._pending_tasks = deque()
        self._current_task = None

    def get_current_task(self):
        return self._current_task

    def _do_report_task(self, task, err_msg=""):
        if self._failed_record_count != 0:
            exec_counters = {
                TaskExecCounterKey.FAIL_COUNT: self._failed_record_count
            }
        else:
            exec_counters = None
        self._worker.report_task_result(
            task.task_id, err_msg, exec_counters=exec_counters
        )

    def _log_fail_records(self, task, err_msg):
        task_len = task.end - task.start
        msg = (
            "records ({f}/{t}) failure, possible "
            "in task_id: {task_id} "
            'reason "{err_msg}"'
        ).format(
            task_id=task.task_id,
            err_msg=err_msg,
            f=self._failed_record_count,
            t=task_len,
        )
        logger.warning(msg)

    def report_record_done(self, count, err_msg=""):
        """
        Report the number of records in the latest processed batch,
        so TaskDataService knows if some pending tasks are finished
        and report_task_result to the master.
        Return True if there are some finished tasks, False otherwise.
        """
        self._reported_record_count += count
        if err_msg:
            self._failed_record_count += count

        task = self._pending_tasks[0]
        total_record_num = task.end - task.start
        if self._reported_record_count >= total_record_num:
            if err_msg:
                self._log_fail_records(task, err_msg)

            # Keep popping tasks until the reported record count is less
            # than the size of the current data since `batch_size` may be
            # larger than `task.end - task.start`
            with self._lock:
                while self._pending_tasks and self._reported_record_count >= (
                    self._pending_tasks[0].end - self._pending_tasks[0].start
                ):
                    task = self._pending_tasks[0]
                    self._reported_record_count -= task.end - task.start
                    self._pending_tasks.popleft()
                    self._do_report_task(task, err_msg)
                    self._failed_record_count = 0
                if self._pending_tasks:
                    self._current_task = self._pending_tasks[0]
            return True
        return False

    def get_dataset_gen(self, task):
        """
        If a task exists, this creates a generator, which could be used to
        creating a `tf.data.Dataset` object in further.
        """
        if not task:
            return None
        tasks = [task]

        def gen():
            for task in tasks:
                for data in self.data_reader.read_records(task):
                    if data:
                        yield data

        return gen

    def get_save_model_task_and_dataset(self):
        if not self._pending_save_model_task:
            return None, None

        task = self._pending_save_model_task
        self._pending_save_model_task = None

        gen = self.get_dataset_gen(task)
        dataset = tf.data.Dataset.from_generator(
            gen, self.data_reader.records_output_types
        )
        return task, dataset

    def get_dataset(self):
        """
        If there's more data, this creates a `tf.data.Dataset` object.
        Otherwise, this returns `None`.
        """
        if self._pending_dataset:
            if self._pending_tasks:
                logger.error(
                    "Cannot get new dataset when there are pending tasks"
                )
                return None
            self._reset()
            # We use a task to perform warm-up for data reader in order
            # to collect useful metadata. Note that we only performs
            # data fetching for this task and `break` instantly to make
            # sure `read_records()` is executed without iterating all the
            # records so this should not be time consuming.
            if self._warm_up_task is None and not self._has_warmed_up:
                task = self._worker.get_task()
                self._warm_up_task = task
                for _ in self.data_reader.read_records(task):
                    break
                self._has_warmed_up = True

            t = threading.Thread(target=self._get_records_fn)
            t.daemon = True
            t.start()

            ds = tf.data.Dataset.from_generator(
                self._gen, self.data_reader.records_output_types
            )
            self._pending_dataset = False
            return ds
        else:
            return None

    def _get_records_fn(self):
        while True:
            # Make sure we also generate data from the warm-up task.
            if self._warm_up_task is not None and self._has_warmed_up:
                task = self._warm_up_task
                self._warm_up_task = None
            else:
                task = self._worker.get_task()
            if not task.shard_name:
                if task.type == elasticdl_pb2.WAIT:
                    self._pending_dataset = True
                    logger.info(
                        "Finish current dataset, maybe more data later"
                    )
                else:
                    logger.info("No more task, stopping")
                break
            with self._lock:
                if task.type == elasticdl_pb2.SAVE_MODEL:
                    self._pending_save_model_task = task
                    continue

                self._pending_tasks.append(task)
                if len(self._pending_tasks) == 1:
                    self._current_task = task

            records = self.data_reader.read_records(task)
            self._records_queue.put(records)

    def _gen(self):
        """
        A generator supports the iter() protocol (e.g. a generator function),
        used to create a `tf.data.Dataset` object from a list of tasks.
        """
        while True:
            records = self._records_queue.get()
            for data in self.data_reader.preprocess_records(records):
                yield data
