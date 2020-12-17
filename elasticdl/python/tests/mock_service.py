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

from concurrent import futures

import grpc
from google.protobuf import empty_pb2

from elasticai_api.proto import elasticai_api_pb2
from elasticdl.proto import elasticdl_pb2_grpc
from elasticdl.python.common.grpc_utils import find_free_port


class MockMasterService(elasticdl_pb2_grpc.MasterServicer):
    def report_evaluation_metrics(self, request, _):
        return empty_pb2.Empty()

    def report_task_result(self, request, _):
        return empty_pb2.Empty()

    def get_task(self, request, _):
        return elasticai_api_pb2.Task()


def _server(server_instance=MockMasterService):
    """Launch a master servicer instance.
    """
    port = find_free_port()
    master_servicer = server_instance()
    svr = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    elasticdl_pb2_grpc.add_MasterServicer_to_server(master_servicer, svr)
    svr.add_insecure_port("[::]:{}".format(port))
    svr.start()
    return svr, port
