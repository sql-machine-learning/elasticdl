import threading
import time
from threading import Thread

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.evaluation_utils import EvaluationMetrics
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.common.tensor import tensor_pb_to_ndarray


class EvaluationJob(object):
    """Representation of an evaluation job"""

    def __init__(self, metrics_dict, model_version, total_tasks=-1):
        """
        Args:
            metrics_dict: A python dictionary. If model has only one output,
                `metrics_dict` is a dictionary of `{metric_name: metric}`,
                i.e. `{"acc": tf.keras.metrics.Accuracy()}`.
                If model has multiple outputs, `metric_dict` is a dictionary of
                `{output_name: {metric_name: metric}}`,
                i.e. `{
                    "output_a": {"acc": tf.keras.metrics.Accuracy()},
                    "output_b": {"auc": tf.keras.metrics.AUC()},
                }`. Note that for model with multiple outputs, each metric
                only uses one output.
            model_version: The version of the model to be evaluated.
            total_tasks: The number of evaluation tasks.
        """

        self.model_version = model_version
        self._total_tasks = total_tasks
        self._completed_tasks = 0
        self.evaluation_metrics = EvaluationMetrics(metrics_dict)

    def complete_task(self):
        self._completed_tasks += 1

    def finished(self):
        return self._completed_tasks >= self._total_tasks

    def report_evaluation_metrics(self, model_outputs_pb, labels):
        labels = tensor_pb_to_ndarray(labels)
        model_outputs = {}
        for tensor_pb in model_outputs_pb:
            key = tensor_pb.name
            model_outputs[key] = tensor_pb_to_ndarray(tensor_pb)
        self.evaluation_metrics.update_evaluation_metrics(
            model_outputs, labels
        )


class _EvaluationTrigger(Thread):
    """A trigger which generates evaluation tasks periodically"""

    def __init__(self, eval_service, start_delay_secs, throttle_secs):
        Thread.__init__(self)
        self._eval_service = eval_service
        self._stopper = threading.Event()
        self._throttle_secs = throttle_secs
        self._eval_min_time = time.time() + start_delay_secs

    def stop(self):
        self._stopper.set()

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

        while not self._stopper.is_set():
            time_now = time.time()
            if self._wait_enough_time(time_now, previous_round_start_secs):
                # Time is up, add an evaluation task
                self._eval_service.add_evaluation_task(is_time_based_eval=True)
                previous_round_start_secs = time_now
            time.sleep(5)


class EvaluationService(object):
    """Evaluation service"""

    def __init__(
        self,
        tensorboard_service,
        task_d,
        start_delay_secs,
        throttle_secs,
        eval_steps,
        eval_only,
        eval_metrics_fn,
    ):
        self._tensorboard_service = tensorboard_service
        self._task_d = task_d
        self._lock = threading.Lock()
        self._eval_job = None
        self.trigger = _EvaluationTrigger(
            self, start_delay_secs, throttle_secs
        )
        self._time_based_eval = throttle_secs > 0
        self._eval_steps = eval_steps
        self._eval_checkpoint_versions = []
        self._last_eval_checkpoint_version = -1
        self._eval_only = eval_only
        self._eval_metrics_fn = eval_metrics_fn

    def start(self):
        if self._time_based_eval and not self._eval_only:
            self.trigger.start()

    def stop(self):
        if self._time_based_eval and not self._eval_only:
            self.trigger.stop()

    def set_master_servicer(self, master_servicer):
        self._master_servicer = master_servicer

    def init_eval_only_job(self, num_task):
        self._eval_job = EvaluationJob(self._eval_metrics_fn(), -1, num_task)

    def add_evaluation_task(
        self, is_time_based_eval, master_locking=True, model_version=None
    ):
        """
        Add evaluation task with current model_version.
        """
        # Do not create time-based eval after all tasks are done
        if is_time_based_eval and self._task_d.finished():
            return
        if not model_version:
            model_version = self._master_servicer.get_model_version()
        if model_version == self._last_eval_checkpoint_version:
            return

        checkpoint_version = model_version
        with self._lock:
            self._eval_checkpoint_versions.append(checkpoint_version)
        self._last_eval_checkpoint_version = checkpoint_version
        self.try_to_create_new_job()

    def try_to_create_new_job(self):
        """
        Add eval task into task dispatcher if current eval_job is done
        and there are pending eval tasks
        """
        with self._lock:
            if self._eval_job is None and self._eval_checkpoint_versions:
                checkpoint_version = self._eval_checkpoint_versions.pop(0)
                self._task_d.create_tasks(
                    elasticdl_pb2.EVALUATION, checkpoint_version
                )
                task_count = len(self._task_d._eval_todo)
                if self._eval_job is None:
                    self._eval_job = EvaluationJob(
                        self._eval_metrics_fn(), checkpoint_version, task_count
                    )
                else:
                    self._eval_job.model_version = checkpoint_version
                    self._eval_job._total_tasks = task_count
                    self._eval_job.reset_metric_states()
                return True
        return False

    def add_evaluation_task_if_needed(self, master_locking, model_version):
        """
        Add step-based evaluation task
        """
        if not model_version:
            model_version = self._master_servicer.get_model_version()
        if (
            self._eval_steps
            and model_version % self._eval_steps == 0
            and model_version > self._last_eval_checkpoint_version
        ):
            self.add_evaluation_task(
                is_time_based_eval=False,
                master_locking=master_locking,
                model_version=model_version,
            )

    def report_evaluation_metrics(self, model_outputs, labels):
        if self._eval_job is None:
            return False
        with self._lock:
            return self._eval_job.report_evaluation_metrics(
                model_outputs, labels
            )

    def complete_task(self):
        self._eval_job.complete_task()
        if self._eval_job.finished():
            evaluation_metrics = (
                self._eval_job.evaluation_metrics.get_evaluation_summary()
            )
            if self._tensorboard_service and evaluation_metrics:
                self._tensorboard_service.write_dict_to_summary(
                    evaluation_metrics, version=self._eval_job.model_version
                )
            logger.info(
                "Evaluation metrics[v=%d]: %s"
                % (
                    self._eval_job.model_version
                    if self._eval_job.model_version >= 0
                    else self._master_servicer.get_model_version(),
                    str(evaluation_metrics),
                )
            )
            if not self._eval_only:
                # delete checkpoint file
                self._eval_job = None
                # create new eval job if possible
                self.try_to_create_new_job()
            return evaluation_metrics
