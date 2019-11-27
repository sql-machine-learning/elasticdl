import threading

from google.protobuf import empty_pb2

from elasticdl.proto import elasticdl_pb2, elasticdl_pb2_grpc
from elasticdl.python.common.log_utils import default_logger as logger


class MasterServicer(elasticdl_pb2_grpc.MasterServicer):
    """Master service implementation"""

    def __init__(
        self, minibatch_size, task_d, evaluation_service,
    ):
        # TODO: group params together into a single object.
        self._task_d = task_d
        self._lock = threading.Lock()
        self._minibatch_size = minibatch_size
        self._version = 0

        self._evaluation_service = evaluation_service
        if evaluation_service:
            evaluation_service.set_master_servicer(self)

    @staticmethod
    def var_name_encode(name):
        return name.replace(":", "-")

    def get_model_version(self):
        return self._version

    def GetTask(self, request, _):
        res = elasticdl_pb2.Task()
        res.model_version = self._version
        res.minibatch_size = self._minibatch_size
        if request.task_type == elasticdl_pb2.EVALUATION:
            task_id, task = self._task_d.get_eval_task(request.worker_id)
        else:
            task_id, task = self._task_d.get(request.worker_id)

        if task:
            res.task_id = task_id
            res.shard_name = task.shard_name
            res.start = task.start
            res.end = task.end
            res.type = task.type
            for k, v in task.extended_config.items():
                res.extended_config[k] = v

            # For evaluation task, it will use the fixed version model
            if task.type == elasticdl_pb2.EVALUATION:
                res.model_version = task.model_version
        elif (not self._task_d.finished()) or (
            self._task_d.invoke_deferred_callback()
        ):
            # If the todo and doing tasks are not empty,
            # Otherwise if the callback list is not empty,
            # we are trying to pop and invoke the callback.
            # Then the master tells the worker to wait
            # in case of new tasks later.
            res.type = elasticdl_pb2.WAIT

        return res

    def ReportTaskResult(self, request, _):
        if request.err_message:
            logger.warning("Worker reported error: " + request.err_message)
            self._task_d.report(request, False)
        else:
            self._task_d.report(request, True)
        return empty_pb2.Empty()

    def ReportEvaluationMetrics(self, request, _):
        self._evaluation_service.report_evaluation_metrics(
            request.model_outputs, request.labels
        )
        return empty_pb2.Empty()

    def ReportVersion(self, request, _):
        self._version = request.model_version
        if self._evaluation_service:
            self._evaluation_service.add_evaluation_task_if_needed(
                master_locking=False, model_version=request.model_version
            )
        return empty_pb2.Empty()
