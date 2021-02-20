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

import collections
from contextlib import closing

import recordio

RecordBlock = collections.namedtuple("RecordBlock", ("name", "start", "end"))


class RecordIOReader(object):
    def __init__(self, recordio_files):
        self._init_file_record_count(recordio_files)

    def _init_file_record_count(self, recordio_files):
        self._data_blocks = []
        start = 0
        for file_path in recordio_files:
            with closing(recordio.Index(file_path)) as rio:
                num_records = rio.num_records()
                end = start + num_records
                self._data_blocks.append(RecordBlock(file_path, start, end))
                start = end

    def read_records(self, start, end):
        target_files = self._get_record_file(start, end)
        for file_path, start, count in target_files:
            with closing(recordio.Scanner(file_path, start, count)) as reader:
                while True:
                    record = reader.record()
                    if record:
                        yield record
                    else:
                        break

    def _get_record_file(self, start, end):
        """The block ranges in data_blocks are sorted in
        increasing order. For example,
        blocks are [[0,100),[100, 200),[200,300)]. So we
        can find which block the shard is in by sequential search.
        """
        target_files = []
        for block in self._data_blocks:
            if start < block.end:
                if end < block.end:
                    target_files.append(
                        (block.name, start - block.start, end - start)
                    )
                    break
                else:
                    target_files.append(
                        (block.name, start - block.start, block.end - start)
                    )
                    start = block.end
        return target_files
