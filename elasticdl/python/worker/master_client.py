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

from elasticdl.proto import elasticdl_pb2, elasticdl_pb2_grpc
from elasticdl.python.common.tensor_utils import serialize_ndarray


class MasterClient:
    """MasterClient provides some APIs connect with the master
    service via gRPC call.

    Usage:
        channel = elasticdl.python.common.grpc_utils.build_channel(
            "localhost:50001"
        )
        mc = MasterClient(channel, work_id=0)
        # get task unit from master service
        mc.get_task(...)
    """

    def __init__(self, channel, worker_id):
        """Initialize a master client.
        Args:
            channel: grpc.Channel
            the gRPC channel object connects to master gRPC server.

            worker_id: int
            the unique and ordered worker ID assigned
            by elasticdl command-line.
        """
        self._stub = elasticdl_pb2_grpc.MasterStub(channel)
        self._worker_id = worker_id

    def get_task(self, task_type=None):
        """Get a task from master.

        Args:
            task_type: elasticdl_pb.TaskType
            the training phase, c.f. /elasticdl/proto/elasticdl.proto

        Returns:
            the task unit assigned by master,
            c.f. /elasticdl/proto/elasticdl.proto
        """

        req = elasticai_api_pb2.GetTaskRequest()
        req.worker_id = self._worker_id
        if task_type is not None:
            req.task_type = task_type

        try:
            res = self._stub.get_task(req)
        except Exception:
            # the master node would stop the gRPC service if no more tasks.
            # And this will result a gRPC call exception.
            res = elasticai_api_pb2.Task()
        return res

    def report_task_result(self, task_id, err_msg, exec_counters=None):
        """Report task result to master.

        Args:
          task_id: int
          the task ID assigned by master

          err_msg: string
          the error message on training.

          exec_counters: dict
          statistics of the task being executed.
        """

        report = elasticai_api_pb2.ReportTaskResultRequest()
        report.task_id = task_id
        report.err_message = err_msg
        if isinstance(exec_counters, dict):
            report.exec_counters.update(exec_counters)
        return self._stub.report_task_result(report)

    def report_evaluation_metrics(self, model_outputs, labels):
        """Report evaluation metrics to master.

        Args:
            model_outputs: dict
            the evaluation result on training.

            labels: numpy array
            the labels on training dataset.
        """
        req = elasticdl_pb2.ReportEvaluationMetricsRequest()
        for name, output in model_outputs.items():
            output = np.concatenate(output)
            serialize_ndarray(output, req.model_outputs[name])
        labels = np.concatenate(labels)
        serialize_ndarray(labels, req.labels)
        req.worker_id = self._worker_id
        self._stub.report_evaluation_metrics(req)

    def get_model_version(self):
        return self._stub.get_model_version()

    def get_comm_rank(self):
        req = elasticai_api_pb2.GetCommRankRequest()
        req.worker_id = self._worker_id
        return self._stub.get_comm_rank(req)
