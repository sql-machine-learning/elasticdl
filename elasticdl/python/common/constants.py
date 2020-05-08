class GRPC(object):
    # gRPC limits the size of message by default to 4MB.
    # It's too small to send model parameters.
    MAX_SEND_MESSAGE_LENGTH = 256 * 1024 * 1024
    MAX_RECEIVE_MESSAGE_LENGTH = 256 * 1024 * 1024


class InstanceManagerStatus(object):
    PENDING = "Pending"
    RUNNING = "Running"
    FINISHED = "Finished"


class MaxComputeConfig(object):
    PROJECT_NAME = "MAXCOMPUTE_PROJECT"
    ACCESS_ID = "MAXCOMPUTE_AK"
    ACCESS_KEY = "MAXCOMPUTE_SK"
    ENDPOINT = "MAXCOMPUTE_ENDPOINT"


class JobType(object):
    TRAINING_ONLY = "training_only"
    EVALUATION_ONLY = "evaluation_only"
    PREDICTION_ONLY = "prediction_only"
    TRAINING_WITH_EVALUATION = "training_with_evaluation"


class Mode(object):
    TRAINING = "training"
    EVALUATION = "evaluation"
    PREDICTION = "prediction"


class Redis(object):
    MAX_COMMAND_RETRY_TIMES = 10


class MetricsDictKey(object):
    MODEL_OUTPUT = "output"
    LABEL = "label"


class DistributionStrategy(object):
    LOCAL = "Local"
    PARAMETER_SERVER = "ParameterServerStrategy"
    ALLREDUCE = "AllreduceStrategy"


class SaveModelConfig(object):
    SAVED_MODEL_PATH = "saved_model_path"


class TaskExecCounterKey(object):
    FAIL_COUNT = "fail_count"


class CollectiveCommunicatorStatus(object):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class PodStatus(object):
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    RUNNING = "Running"
    FINISHED = "Finished"
    PENDING = "Pending"


class ReaderType(object):
    CSV_READER = "CSV"
    ODPS_READER = "ODPS"
    RECORDIO_READER = "RecordIO"


class BashCommandTemplate(object):
    REDIRECTION = " 2>&1 | tee {}"
