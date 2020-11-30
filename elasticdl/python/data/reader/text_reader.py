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

import csv
import linecache

import numpy as np
import tensorflow as tf

from elasticdl.python.data.reader.data_reader import (
    AbstractDataReader,
    Metadata,
)


class TextDataReader(AbstractDataReader):
    """This reader is used to create shards for a file and
    read records from the shard.
    """

    def __init__(self, **kwargs):
        """
        Args:
            kwargs should contains "filename" and "records_per_task".
        """
        AbstractDataReader.__init__(self, **kwargs)
        self._kwargs = kwargs
        self._filename = self._kwargs["filename"]

    def read_records(self, task, shuffle=False):
        records = linecache.getlines(task.shard.name)[
            task.shard.start : task.shard.end
        ]
        if shuffle:
            np.random.shuffle(records)

        for record in records:
            yield record

    def create_shards(self):
        shard_name_prefix = self._kwargs["filename"] + ":shard_"
        size = self.get_size()
        records_per_task = self._kwargs["records_per_task"]
        shards = {}
        num_shards = size // records_per_task
        start_ind = 0
        for shard_id in range(num_shards):
            shards[shard_name_prefix + str(shard_id)] = (
                start_ind,
                records_per_task,
            )
            start_ind += records_per_task
        num_records_left = size % records_per_task
        if num_records_left != 0:
            shards[shard_name_prefix + str(num_shards)] = (
                start_ind,
                num_records_left,
            )
        return shards

    def get_size(self):
        with open(self._filename) as file:
            reader = csv.reader(file)
            line_num = len(list(reader))
            return line_num

    @property
    def records_output_types(self):
        return tf.string

    @property
    def metadata(self):
        return Metadata(column_names=None)
