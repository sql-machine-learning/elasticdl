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
from concurrent import futures

import grpc
from google.protobuf import empty_pb2

from elasticai_api.common.constants import GRPC, TrainingLoopStatus
from elasticai_api.proto import elasticai_api_pb2, elasticai_api_pb2_grpc
from elasticdl.proto import elasticdl_pb2_grpc
from elasticdl.python.common.log_utils import default_logger as logger


def create_master_service(
    port, task_manager, pod_manager, rendezvous_server, evaluation_service,
):
    """Create GRPC server
    """
    logger.info("Creating master service")
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=64),
        options=[
            ("grpc.max_send_message_length", GRPC.MAX_SEND_MESSAGE_LENGTH),
            (
                "grpc.max_receive_message_length",
                GRPC.MAX_RECEIVE_MESSAGE_LENGTH,
            ),
        ],
    )
    master_servicer = MasterServicer(
        task_manager=task_manager,
        instance_manager=pod_manager,
        rendezvous_server=rendezvous_server,
        evaluation_service=evaluation_service,
    )
    elasticai_api_pb2_grpc.add_MasterServicer_to_server(
        master_servicer, server
    )
    elasticdl_pb2_grpc.add_TrainLoopMasterServicer_to_server(
        master_servicer, server
    )
    server.add_insecure_port("[::]:{}".format(port))
    logger.info("The port of the master server is: %d", port)

    return server


class MasterServicer(
    elasticai_api_pb2_grpc.MasterServicer,
    elasticdl_pb2_grpc.TrainLoopMasterServicer,
):
    """Master service implementation"""

    def __init__(
        self,
        task_manager,
        instance_manager,
        rendezvous_server=None,
        evaluation_service=None,
    ):
        # TODO: group params together into a single object.
        self._task_manager = task_manager
        self._instance_manager = instance_manager
        self._rendezvous_server = rendezvous_server
        self._evaluation_service = evaluation_service
        if self._evaluation_service:
            self._evaluation_service.set_model_version_fn(
                self.get_model_version
            )
        self._lock = threading.Lock()
        self._version = 0

    def get_model_version(self):
        return self._version

    def get_task(self, request, _):
        shard = elasticai_api_pb2.Shard()
        res = elasticai_api_pb2.Task(shard=shard)
        res.model_version = self._version
        if request.task_type == elasticai_api_pb2.EVALUATION:
            task_id, task = self._task_manager.get_eval_task(request.worker_id)
        else:
            task_id, task = self._task_manager.get(request.worker_id)

        if task:
            res.task_id = task_id
            res.type = task.type
            res.shard.name = task.shard.name
            res.shard.start = task.shard.start
            res.shard.end = task.shard.end
            res.shard.indices.extend(task.shard.indices)
            for k, v in task.extended_config.items():
                res.extended_config[k] = v

            # For evaluation task, it will use the fixed version model
            if task.type == elasticai_api_pb2.EVALUATION:
                res.model_version = task.model_version
        elif (not self._task_manager.finished()) or (
            self._task_manager.invoke_deferred_callback()
        ):
            # If the todo and doing tasks are not empty,
            # Otherwise if the callback list is not empty,
            # we are trying to pop and invoke the callback.
            # Then the master tells the worker to wait
            # in case of new tasks later.
            if self._rendezvous_server:
                # If there is no more task, master only send wait task to
                # the last worker and other workers exit.
                if len(self._instance_manager.get_alive_workers()) == 1:
                    res.type = elasticai_api_pb2.WAIT
            else:
                res.type = elasticai_api_pb2.WAIT
        with self._lock:
            self._task_manager.reset_worker_start_task_time(request.worker_id)
        return res

    def report_task_result(self, request, _):
        if self._task_manager.support_fault_tolerance:
            if request.err_message:
                logger.warning("Worker reported error: " + request.err_message)
                self._task_manager.report(request, False)
            else:
                complete_time, task, worker_id = self._task_manager.report(
                    request, True
                )
                if task:
                    with self._lock:
                        self._task_manager.reset_worker_start_task_time(
                            worker_id
                        )
                        if task.type in [
                            elasticai_api_pb2.TRAINING,
                            elasticai_api_pb2.EVALUATION,
                        ]:
                            self._task_manager.record_task_completed_time(
                                task.type, complete_time
                            )
        return empty_pb2.Empty()

    def report_evaluation_metrics(self, request, _):
        with self._lock:
            self._task_manager.reset_worker_start_task_time(request.worker_id)
        self._evaluation_service.report_evaluation_metrics(
            request.model_outputs, request.labels
        )
        return empty_pb2.Empty()

    def report_version(self, request, _):
        self._version = request.model_version
        if self._evaluation_service:
            self._evaluation_service.add_evaluation_task_if_needed(
                model_version=request.model_version
            )
        return empty_pb2.Empty()

    def report_training_params(self, request, _):
        self._task_manager.set_training_params(
            request.batch_size,
            request.num_epochs,
            request.dataset_size,
            request.shuffle,
            request.shuffle_shards,
        )
        return empty_pb2.Empty()

    def get_comm_rank(self, request, _):
        worker_host = request.worker_host
        res = elasticai_api_pb2.GetCommRankResponse()
        res.rank_id = self._rendezvous_server.get_worker_host_rank(worker_host)
        res.world_size = self._rendezvous_server.get_size()
        res.rendezvous_id = self._rendezvous_server.get_rendezvous_id()
        res.rendezvous_port = self._rendezvous_server.get_rendezvous_port()
        return res

    def report_training_loop_status(self, request, _):
        training_loop_status = request.status
        if not self._rendezvous_server:
            logger.warning("The rendezvous server does not exit")
            return empty_pb2.Empty()
        if training_loop_status == TrainingLoopStatus.START:
            self._rendezvous_server.add_worker(request.worker_host)
        elif training_loop_status == TrainingLoopStatus.END:
            self._rendezvous_server.remove_worker(request.worker_host)
        return empty_pb2.Empty()
