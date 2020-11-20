# Copyright 2020 The ElasticDL Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.evaluation_utils import EvaluationMetrics
from elasticdl.python.common.log_utils import default_logger as logger
from elasticdl.python.common.tensor_utils import pb_to_ndarray


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
        labels = pb_to_ndarray(labels)
        model_outputs = {}
        for name, tensor_pb in model_outputs_pb.items():
            model_outputs[name] = pb_to_ndarray(tensor_pb)
        self.evaluation_metrics.update_evaluation_metrics(
            model_outputs, labels
        )


class EvaluationService(object):
    """Evaluation service"""

    def __init__(
        self, task_d, eval_steps, eval_only, eval_metrics_fn,
    ):
        self._task_d = task_d
        self._lock = threading.Lock()
        self._eval_job = None
        self._eval_steps = eval_steps
        self._eval_checkpoint_versions = []
        self._last_eval_checkpoint_version = -1
        self._eval_only = eval_only
        self._eval_metrics_fn = eval_metrics_fn

    def set_master_servicer(self, master_servicer):
        self._master_servicer = master_servicer

    def init_eval_only_job(self, num_task):
        self._eval_job = EvaluationJob(self._eval_metrics_fn(), -1, num_task)

    def add_evaluation_task(self, model_version=None):
        """
        Add evaluation task with current model_version.
        """
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

    def add_evaluation_task_if_needed(self, model_version):
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
            self.add_evaluation_task(model_version=model_version,)

    def report_evaluation_metrics(self, model_outputs, labels):
        if self._eval_job is None:
            return False
        with self._lock:
            return self._eval_job.report_evaluation_metrics(
                model_outputs, labels
            )

    def complete_task(self):
        if self._eval_job is None:
            return
        self._eval_job.complete_task()
        if self._eval_job.finished():
            evaluation_metrics = (
                self._eval_job.evaluation_metrics.get_evaluation_summary()
            )
            logger.info(
                "Evaluation metrics[step=%d]: %s"
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
