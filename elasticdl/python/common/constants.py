class GRPC(object):
    # gRPC limits the size of message by default to 4MB.
    # It's too small to send model parameters.
    MAX_SEND_MESSAGE_LENGTH = 256 * 1024 * 1024
    MAX_RECEIVE_MESSAGE_LENGTH = 256 * 1024 * 1024


class WorkerManagerStatus(object):
    PENDING = "Pending"
    RUNNING = "Running"
    FINISHED = "Finished"


class ODPSConfig(object):
    PROJECT_NAME = "ODPS_PROJECT_NAME"
    ACCESS_ID = "ODPS_ACCESS_ID"
    ACCESS_KEY = "ODPS_ACCESS_KEY"
    ENDPOINT = "ODPS_ENDPOINT"


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
    PARAMETER_SERVER = "ParameterServerStrategy"
