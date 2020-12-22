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

from elasticai_api.common.master_client import MasterClient as BaseMasterClient
from elasticdl.proto import elasticdl_pb2, elasticdl_pb2_grpc
from elasticdl.python.common.tensor_utils import serialize_ndarray


class MasterClient(BaseMasterClient):
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
        super(MasterClient, self).__init__(
            channel=channel, worker_id=worker_id
        )
        self._train_loop_stub = elasticdl_pb2_grpc.TrainLoopMasterStub(channel)

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
        self._train_loop_stub.report_evaluation_metrics(req)
