# Copyright 2021 The ElasticDL Authors. All rights reserved.
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

import tensorflow as tf

from elasticai_api.util.log_utils import default_logger as logger


class ElasticDataShardReportHook(tf.train.SessionRunHook):
    def __init__(self, data_shard_service) -> None:
        self._data_shard_service = data_shard_service

    def after_run(self, run_context, run_values):
        try:
            self._data_shard_service.report_batch_done()
        except Exception as ex:
            logger.error("elastic_ai: report batch done failed: %s", ex)
