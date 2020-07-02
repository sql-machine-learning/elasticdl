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
    TUNNEL_ENDPOINT = "MAXCOMPUTE_TUNNEL_ENDPOINT"


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
    SET_PIPEFAIL = "set -o pipefail;"
