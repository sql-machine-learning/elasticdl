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

import numpy as np
from tensorflow.python.keras import metrics as metrics_module

from elasticdl.python.common.constants import MetricsDictKey


class EvaluationMetrics(object):
    """Evaluation Metrics"""

    def __init__(self, metrics_dict):
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
        """
        self._init_metrics_dict(metrics_dict)

    def _init_metrics_dict(self, metrics_dict):
        if not metrics_dict:
            raise ValueError(
                "Evaluation metrics dictionary must not be empty."
            )
        first_metrics = list(metrics_dict.values())[0]
        if isinstance(first_metrics, dict):
            self._model_have_multiple_outputs = True
            self._metrics_dict = metrics_dict
        else:
            # When model has only one output, save it in a dict in order to
            # keep the same data structure as the `metrics_dict` when model
            # has multiple outputs.
            self._model_have_multiple_outputs = False
            self._metrics_dict = {MetricsDictKey.MODEL_OUTPUT: metrics_dict}
        for output_name, metrics in self._metrics_dict.items():
            for metric_name, metric in metrics.items():
                if not isinstance(metric, metrics_module.Metric):
                    # `tf.keras.metrics.MeanMetricWrapper` wraps stateless
                    # functions into `tf.keras.metrics.Metric` instance.
                    metrics[metric_name] = metrics_module.MeanMetricWrapper(
                        metric, name=metric_name
                    )

    def update_evaluation_metrics(self, model_outputs, labels):
        for key in model_outputs:
            metrics = self._metrics_dict.get(key, {})
            if not metrics:
                continue
            outputs = model_outputs.get(key)
            for metric_inst in metrics.values():
                self._update_metric_by_small_chunk(
                    metric_inst, labels, outputs
                )

    def get_evaluation_summary(self):
        if self._model_have_multiple_outputs:
            return {
                output_name: {
                    metric_name: metric_inst.result().numpy()
                    for metric_name, metric_inst in metrics.items()
                }
                for output_name, metrics in self._metrics_dict.items()
            }
        return {
            metric_name: metric_inst.result().numpy()
            for metric_name, metric_inst in self._metrics_dict[
                MetricsDictKey.MODEL_OUTPUT
            ].items()
        }

    def reset_metric_states(self):
        """Resets all of the metric state variables."""
        for metrics in self._metrics_dict.values():
            for metric_inst in metrics.values():
                metric_inst.reset_states()

    @staticmethod
    def _update_metric_by_small_chunk(
        metric, labels, outputs, chunk_length=500
    ):
        """The metric updates state in a thread launched by grpc. The memory will
        increase greatly if we update the metric with large size outputs. So
        we split the outputs and labels to small chunks then update the metric
        with those small chunks. The [issue 35044](https://github.com/
        tensorflow/tensorflow/issues/35044) has been submitted to tensorflow.
        """
        chunk_boundaries = np.asarray(range(0, len(labels), chunk_length))
        label_chunks = np.array_split(labels, chunk_boundaries)
        output_chunks = np.array_split(outputs, chunk_boundaries)
        for label, output in zip(label_chunks, output_chunks):
            metric.update_state(label, output)
