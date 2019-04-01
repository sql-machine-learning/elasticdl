
from proto import master_pb2
from proto import master_pb2_grpc


class MasterServicer(master_pb2_grpc.MasterServicer):
    """Master service implementation"""

    def __init__(self, logger):
        self.logger = logger
        # TODO: random initialization
        self._model = {}
        self._version = 0
        self._gradient_sum = {}

    def GetTask(self, request, context):
        # TODO: implent task queues. Return an empty task for now.
        res = master_pb2.Task()
        res.shard_file_name = ""
        res.model_version = self._version
        return res

    def GetModel(self, request, context):
        if request.min_version > self._version:
            err_msg = (
                "Requested version %d not available yet, current version: %d"
                % (request.min_version, self._version)
            )
            self.logger.warning(err_msg)
            raise ValueError(err_msg)

        res = master_pb2.Model()
        res.version = self._version
        # TODO: convert self._model to tensor.
        return res

    def ReportTaskResult(self, request, context):
        if request.model_version > self._version:
            err_msg = "Model version %d out of range, current version: %d" % (
                request.model_version,
                self._version,
            )
            self.logger.warning(err_msg)
            raise ValueError(err_msg)

        res = master_pb2.ReportTaskResultReply()
        if request.model_version < self._version:
            self.logger.warning(
                "Task result for outdated version %d dropped",
                request.model_version,
            )
            res.accepted = False
            res.model_version = self._version
            return res

        if request.err_message:
            self.logger.warning("Worker error: %s" % request.err_message)
            res.accepted = False
            res.model_version = self._version
            return res

        # TODO: Update task queue with task_id
        # TODO: accumulate gradient
        # TODO: update model
        res.accepted = True
        res.model_version = self._version
        return res
