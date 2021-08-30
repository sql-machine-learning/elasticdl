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
import unittest

from elasticai_api.common.data_shard_service import build_data_shard_service


class DataShardServiceTest(unittest.TestCase):
    def test_data_shard_servie(self):
        data_shard_service = build_data_shard_service(64, dataset_name="test")
        shard = data_shard_service.fetch_shard()
        self.assertEqual(shard.start, 0)
        self.assertEqual(shard.end, 100)
        self.assertEqual(data_shard_service.get_current_epoch(), 0)


if __name__ == "__main__":
    unittest.main()