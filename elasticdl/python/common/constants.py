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


class PodManagerStatus(object):
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


class MetricsDictKey(object):
    MODEL_OUTPUT = "output"
    LABEL = "label"


class CollectiveCommunicatorStatus(object):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class PodStatus(object):
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    RUNNING = "Running"
    FINISHED = "Finished"
    PENDING = "Pending"
    INITIAL = "Initial"
    DELETED = "Deleted"


class ReaderType(object):
    CSV_READER = "CSV"
    ODPS_READER = "ODPS"
    RECORDIO_READER = "RecordIO"


class Initializer(object):
    UNIFORM = "uniform"


class WorkerMemoryConfig(object):
    MIN_MEMORY = 4096  # 4096Mi, 4Gi
    MAX_INCREMENTAL_MEMORY = 8192  # 8Gi
    HUGE_RESOURCE_THRESHOLD = 102400  # 100Gi
    ADJUSTMENT_FACTOR = 1.8
    WAIT_CHIEF_WORKER_TIMEOUT_SECS = 1800  # 30min
    WAIT_DATA_SHARD_SERVICE_CREATION_SECS = 600  # 10min
