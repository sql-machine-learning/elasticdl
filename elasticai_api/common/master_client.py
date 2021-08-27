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
from elasticai_api.util.log_utils import default_logger as logger


class MasterClient(object):
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

    def reset_dataset(self, dataset_name):
        """ Reset a dataset

        Args:
            dataset_name: name of the dataset, must not be None.
        """
        if dataset_name:
            req = elasticai_api_pb2.ResetDatasetRequest()
            req.dataset_name = dataset_name
            self._stub.reset_dataset(req)

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

    def get_dataset_task(self, dataset_name):
        """Get a task from master.

        Args:
            dataset_name: string
            the training phase, c.f. /elasticdl/proto/elasticdl.proto

        Returns:
            the task unit assigned by master,
            c.f. /elasticdl/proto/elasticdl.proto
        """

        req = elasticai_api_pb2.GetDatasetTaskRequest()
        req.worker_id = self._worker_id
        req.dataset_name = dataset_name

        try:
            res = self._stub.get_dataset_task(req)
        except Exception as e:
            logger.warning(e)
            # the master node would stop the gRPC service if no more tasks.
            # And this will result a gRPC call exception.
            res = elasticai_api_pb2.Task()
        return res

    def report_task_result(
        self, task_id, err_msg, dataset_name=None, exec_counters=None
    ):
        """Report task result to master.

        Args:
          task_id: int
          the task ID assigned by master

          err_msg: string
          the error message on training.

          exec_counters: dict
          statistics of the task being executed.
        """
        if dataset_name:
            request = elasticai_api_pb2.ReportDatasetTaskResultRequest()
            request.dataset_name = dataset_name
        else:
            request = elasticai_api_pb2.ReportTaskResultRequest()
        request.task_id = task_id

        request.err_message = err_msg
        if isinstance(exec_counters, dict):
            request.exec_counters.update(exec_counters)
        if dataset_name:
            return self._stub.report_dataset_task_result(request)
        else:
            return self._stub.report_task_result(request)

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
        num_minibatches_per_shard=0,
        dataset_name=None,
    ):
        if dataset_name:
            request = elasticai_api_pb2.ReportDatasetShardParamsRequest()
            request.batch_size = batch_size
            request.shuffle = shuffle
            request.shuffle_shards = shuffle_shards
            request.dataset_name = dataset_name
            if num_epochs is not None:
                request.num_epochs = num_epochs
            if dataset_size is not None:
                request.dataset_size = dataset_size
            request.num_minibatches_per_shard = num_minibatches_per_shard
            return self._stub.report_dataset_shard_params(request)
        else:
            request = elasticai_api_pb2.ReportTrainingParamsRequest()
            request.batch_size = batch_size
            request.shuffle = shuffle
            request.shuffle_shards = shuffle_shards
            if num_epochs is not None:
                request.num_epochs = num_epochs
            if dataset_size is not None:
                request.dataset_size = dataset_size
            request.num_minibatches_per_shard = num_minibatches_per_shard
            return self._stub.report_training_params(request)

    def get_shard_checkpoint(self, dataset_name):
        request = elasticai_api_pb2.GetShardCheckpointRequest()
        request.dataset_name = dataset_name if dataset_name else ""
        return self._stub.get_shard_checkpoint(request)

    def report_shard_checkpoint(self, shard_checkpoint):
        request = elasticai_api_pb2.ShardCheckpoint()
        request.content = shard_checkpoint
        return self._stub.report_shard_checkpoint(request)

    def worker_sync(self, sync_name):
        request = elasticai_api_pb2.WorkerSyncRequest()
        request.sync_name = sync_name
        request.worker_id = self._worker_id
        return self._stub.worker_sync(request)

    def wait_worker_sync(self, sync_name, notify):
        request = elasticai_api_pb2.WaitWorkerSyncRequest()
        request.sync_name = sync_name
        request.notify = notify
        return self._stub.wait_worker_sync(request)

    def delete_worker_sync(self, sync_name):
        request = elasticai_api_pb2.DeleteWorkerSyncRequest()
        request.sync_name = sync_name
        request.delete_all = False
        return self._stub.delete_worker_sync(request)

    def delete_all_worker_sync(self):
        request = elasticai_api_pb2.DeleteWorkerSyncRequest()
        request.delete_all = True
        return self._stub.delete_worker_sync(request)

    def report_used_resource(self, memory, cpu_percent):
        request = elasticai_api_pb2.ReportUsedResourceRequest()
        request.memory = memory
        request.cpu_percent = cpu_percent
        request.work_id = self._worker_id
        return self._stub.report_used_resource(request)

    def get_dataset_epoch(self, dataset_name):
        request = elasticai_api_pb2.GetDatasetEpochRequest()
        request.dataset_name = dataset_name if dataset_name else ""
        return self._stub.get_dataset_epoch(request)


class MockedMasterClient(object):
    """MockedMasterClient provides the same API as MasterClient without
    any RPC call.
    """

    def __init__(self, worker_id):
        """Initialize a master client.
        Args:
            worker_id: int
            the unique and ordered worker ID assigned
            by elasticdl command-line.
        """
        self._worker_id = worker_id

    def reset_dataset(self, dataset_name):
        """ Reset a dataset

        Args:
            dataset_name: name of the dataset, must not be None.
        """
        pass

    def get_task(self, task_type=None):
        """Get a task from master.

        Args:
            task_type: elasticdl_pb.TaskType
            the training phase, c.f. /elasticdl/proto/elasticdl.proto

        Returns:
            the task unit assigned by master,
            c.f. /elasticdl/proto/elasticdl.proto
        """
        shard = elasticai_api_pb2.Shard()
        res = elasticai_api_pb2.Task(shard=shard)
        res.shard.start = 0
        res.shard.end = 100
        return res

    def get_dataset_task(self, dataset_name):
        """Get a task from master.

        Args:
            dataset_name: string
            the training phase, c.f. /elasticdl/proto/elasticdl.proto

        Returns:
            the task unit assigned by master,
            c.f. /elasticdl/proto/elasticdl.proto
        """

        shard = elasticai_api_pb2.Shard()
        res = elasticai_api_pb2.Task(shard=shard)
        res.shard.start = 0
        res.shard.end = 100
        res.type = elasticai_api_pb2.TRAINING
        return res

    def report_task_result(
        self, task_id, err_msg, dataset_name=None, exec_counters=None
    ):
        """Report task result to master.

        Args:
          task_id: int
          the task ID assigned by master

          err_msg: string
          the error message on training.

          exec_counters: dict
          statistics of the task being executed.
        """
        return True

    def get_comm_rank(self):
        return 0

    def report_training_loop_status(self, status):
        return True

    def report_training_params(
        self,
        batch_size,
        num_epochs=None,
        dataset_size=None,
        shuffle=False,
        shuffle_shards=False,
        num_minibatches_per_shard=0,
        dataset_name=None,
    ):
        return True

    def query_relaunch_ps_pod(self):
        res = elasticai_api_pb2.QueryRelaunchPSPodResponse()
        return res

    def ready_for_ps_relaunch(self):
        return None

    def get_shard_checkpoint(self, dataset_name):
        res = elasticai_api_pb2.ReportShardCheckpointResponse()
        return res

    def report_shard_checkpoint(self, shard_checkpoint):
        return None

    def worker_sync(self, sync_name):
        res = elasticai_api_pb2.WorkerSyncResponse()
        return res

    def wait_worker_sync(self, sync_name, notify):
        res = elasticai_api_pb2.WorkerSyncResponse()
        return res

    def delete_worker_sync(self, sync_name):
        return None

    def delete_all_worker_sync(self):
        return None

    def report_used_resource(self, memory, cpu_percent):
        return None

    def get_dataset_epoch(self, dataset_name):
        res = elasticai_api_pb2.GetDatasetEpochResponse()
        res.epoch = 0
        return res


def build_master_client():
    master_addr = os.getenv("MASTER_ADDR", "")
    worker_id = int(os.getenv("WORKER_ID", 0))

    if master_addr:
        master_client = MasterClient(build_channel(master_addr), worker_id)
    else:
        master_client = MockedMasterClient(worker_id)

    return master_client


class GlobalMasterClient(object):
    MASTER_CLIENT = build_master_client()
