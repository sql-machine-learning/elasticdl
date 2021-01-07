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

import os

from elasticai_api.proto import elasticai_api_pb2, elasticai_api_pb2_grpc
from elasticai_api.util.grpc_utils import build_channel


def build_master_client():
    master_addr = os.getenv("MASTER_ADDR", "localhost:12345")
    worker_id = int(os.getenv("WORKER_ID", 0))

    master_client = MasterClient(build_channel(master_addr), worker_id)

    return master_client


class MasterClient:
    """MasterClient provides some APIs connect with the master
    service via gRPC call.

    Usage:
        channel = elasticai_api.util.grpc_utils.build_channel(
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
        self._stub = elasticai_api_pb2_grpc.MasterStub(channel)
        self._worker_id = worker_id
        self._worker_host = os.getenv("MY_POD_IP", "localhost")

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

    def get_comm_rank(self):
        req = elasticai_api_pb2.GetCommRankRequest()
        req.worker_host = self._worker_host
        return self._stub.get_comm_rank(req)

    def report_training_loop_status(self, status):
        req = elasticai_api_pb2.ReportTrainingLoopStatusRequest()
        req.worker_host = self._worker_host
        req.status = status
        return self._stub.report_training_loop_status(req)

    def report_training_params(
        self,
        batch_size,
        num_epochs=None,
        dataset_size=None,
        shuffle=False,
        shuffle_shards=False,
    ):
        report = elasticai_api_pb2.ReportTrainingParamsRequest()
        report.batch_size = batch_size
        report.shuffle = shuffle
        report.shuffle_shards = shuffle_shards
        if num_epochs is not None:
            report.num_epochs = num_epochs
        if dataset_size is not None:
            report.dataset_size = dataset_size
        return self._stub.report_training_params(report)
