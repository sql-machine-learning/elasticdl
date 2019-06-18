import logging
import time
import numpy as np

from collections import defaultdict
from threading import Thread

from elasticdl.python.elasticdl.common.ndarray import tensor_to_ndarray


class _EvaluationJob(object):
    """Representation of an evaluation job"""

    def __init__(self, model_version, total_tasks=-1):
        self._model_version = model_version
        self._total_tasks = total_tasks
        self._completed_tasks = 0
        self._evaluation_metrics = defaultdict(list)

    def complete_task(self):
        self._completed_tasks += 1

    def finished(self):
        return self._completed_tasks >= self._total_tasks

    def ok_to_new_job(self, latest_chkp_version):
        return self.finished() and latest_chkp_version > self._model_version

    def report_evaluation_metrics(
        self, evaluation_version, evaluation_metrics
    ):
        if evaluation_version != self._model_version:
            return False
        for k, v in evaluation_metrics.items():
            if v.dim:
                self._evaluation_metrics[k].append(tensor_to_ndarray(v))
        return True

    def get_evaluation_summary(self):
        return {k: np.mean(v) for k, v in self._evaluation_metrics.items()}


class _EvaluationTrigger(Thread):
    """A trigger which generates evaluation tasks periodically"""

    def __init__(self, eval_service, stopped, start_delay_secs, throttle_secs):
        Thread.__init__(self)
        self._eval_service = eval_service
        self._stopped = stopped
        self._throttle_secs = throttle_secs
        self._eval_min_time = time.time() + start_delay_secs

    def _wait_enough_time(self, cur_time_secs, previous_round_start_secs):
        if cur_time_secs < self._eval_min_time:
            return False
        if (
            previous_round_start_secs != -1
            and cur_time_secs - previous_round_start_secs < self._throttle_secs
        ):
            return False
        return True

    def run(self):
        previous_round_start_secs = -1
        while not self._stopped(5):
            time_now = time.time()
            if self._wait_enough_time(time_now, previous_round_start_secs):
                # Time is up, trying to start a new round evaluation
                new_round = self._eval_service.try_to_create_new_round()
                previous_round_start_secs = (
                    time_now if new_round else previous_round_start_secs
                )


class EvaluationService(object):
    """Evaluation service"""

    def __init__(
        self,
        checkpoint_service,
        task_q,
        stopped,
        start_delay_secs,
        throttle_secs,
    ):
        self._logger = logging.getLogger(__name__)
        self._checkpoint_service = checkpoint_service
        self._task_q = task_q
        self._eval_job = None
        self.trigger = _EvaluationTrigger(
            checkpoint_service, stopped, start_delay_secs, throttle_secs
        )

    def start(self):
        self.trigger.start()

    def try_to_create_new_round(self):
        try:
            latest_chkp_version = (
                self._checkpoint_service.get_latest_checkpoint_version()
            )
            if self._eval_job is None or self._eval_job.ok_to_new_job(
                latest_chkp_version
            ):
                tasks = self._task_q.create_evaluation_tasks(
                    latest_chkp_version
                )
                self._eval_job = _EvaluationJob(
                    latest_chkp_version, len(tasks)
                )
                return True
        except Exception as e:
            self._logger.error(
                "Failed to create evaluation tasks: %s" % str(e)
            )

        return False

    def report_evaluation_metrics(
        self, evaluation_version, evaluation_metrics
    ):
        if self._eval_job is None:
            return False
        return self._eval_job.report_evaluation_metrics(
            evaluation_version, evaluation_metrics
        )

    def complete_task(self):
        self._eval_job.complete_task()
        if self._eval_job.finished():
            evaluation_metrics = self._eval_job.get_evaluation_summary()
            self._logger.info("Evaluation metrics: %s" % evaluation_metrics)
