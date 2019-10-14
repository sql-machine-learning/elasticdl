import threading

import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.data.data_reader import create_data_reader
from elasticdl.python.data.dataset_utils import create_dataset_from_tasks


class TaskDataService(object):
    def __init__(
        self, worker, training_with_evaluation, data_reader_params=None
    ):
        self._worker = worker
        self._training_with_evaluation = training_with_evaluation
        self._lock = threading.Lock()
        self._pending_dataset = True
        self._pending_eval_tasks = []
        self._reset()
        if data_reader_params:
            self.data_reader = create_data_reader(
                data_origin=None, **data_reader_params
            )
        else:
            self.data_reader = create_data_reader(data_origin=None)
        self._warm_up_task = None
        self._has_warmed_up = False

    def _reset(self):
        """
        Reset pending tasks and record counts
        """
        self._record_count = 0
        self._reported_record_count = 0
        self._pending_tasks_with_counts = []
        self._current_task = None

    def get_current_task(self):
        return self._current_task

    def report_record_done(self, count, err_msg=""):
        """
        Report the number of records in the latest processed batch,
        so TaskDataService knows if some pending tasks are finished
        and report_task_result to the master.
        self._pending_tasks_with_counts[0][0] is the first pending task,
        self._pending_tasks_with_counts[0][1] is the number of records
        in this task.
        """
        self._reported_record_count += count
        if (
            len(self._pending_tasks_with_counts)
            and self._reported_record_count
            >= self._pending_tasks_with_counts[0][1]
        ):
            with self._lock:
                while (
                    len(self._pending_tasks_with_counts)
                    and self._reported_record_count
                    >= self._pending_tasks_with_counts[0][1]
                ):
                    task, _ = self._pending_tasks_with_counts.pop(0)
                    self._worker.report_task_result(task.task_id, err_msg)
                if len(self._pending_tasks_with_counts):
                    self._current_task = self._pending_tasks_with_counts[0][0]

    def get_evaluation_dataset(self):
        """
        If there are _pending_eval_tasks, return a RecordIO dataset for
        an evaluation task and its corresponding model version, task_id.
        Return None if no _pending_eval_tasks.
        """
        if not self._pending_eval_tasks:
            return None
        shards = []
        task = None
        with self._lock:
            if self._pending_eval_tasks:
                task = self._pending_eval_tasks.pop(0)
                shards.append(task)
        if shards and task:
            return (
                create_dataset_from_tasks(shards, self.data_reader),
                task.model_version,
                task.task_id,
            )
        else:
            return None

    def get_dataset(self):
        """
        Return a RecordIO dataset, or None if no more data.
        """
        if self._pending_dataset:
            if self._pending_tasks_with_counts:
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
            ds = tf.data.Dataset.from_generator(
                self._gen, self.data_reader.records_output_types
            )
            self._pending_dataset = False
            return ds
        else:
            return None

    def _gen(self):
        """
        A generator supports the iter() protocol (e.g. a generator function),
        used to create a `tf.data.Dataset` from a list of tasks.
        """
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
                if (
                    self._training_with_evaluation
                    and task.type == elasticdl_pb2.EVALUATION
                ):
                    self._pending_eval_tasks.append(task)
                    continue
                self._record_count += task.end - task.start
                self._pending_tasks_with_counts.append(
                    (task, self._record_count)
                )
                if len(self._pending_tasks_with_counts) == 1:
                    self._current_task = task
            for data in self.data_reader.read_records(task):
                if data:
                    yield data
