import math
import os
from abc import ABC, abstractmethod
from contextlib import closing

import recordio

from elasticdl.python.common.odps_io import ODPSReader


class AbstractDataReader(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def read_records(self, task):
        """This method will be used in `TaskDataService` to read the records
        based on the information provided for a given task into a Python
        generator/iterator.

        Arguments:
            task: The current `Task` object that provides information on where
                to read the data for this task.
        """
        pass

    @abstractmethod
    def create_shards(self):
        """This method creates the dictionary of shards where the keys
        are the shard names and the values are tuples of the starting
        index and the number of records in each shard.
        """
        pass


class RecordIODataReader(AbstractDataReader):
    def __init__(self, **kwargs):
        AbstractDataReader.__init__(self, **kwargs)
        self._kwargs = kwargs
        if "data_dir" not in self._kwargs:
            raise ValueError("data_dir is required for RecordIODataReader()")

    def read_records(self, task):
        with closing(
            recordio.Scanner(
                task.shard_name, task.start, task.end - task.start
            )
        ) as reader:
            while True:
                record = reader.record()
                if record:
                    yield record
                else:
                    break

    def create_shards(self):
        data_dir = self._kwargs["data_dir"]
        start_ind = 0
        if not data_dir:
            return {}
        f_records = {}
        for f in os.listdir(data_dir):
            p = os.path.join(data_dir, f)
            with closing(recordio.Index(p)) as rio:
                f_records[p] = (start_ind, rio.num_records())
        return f_records


class ODPSDataReader(AbstractDataReader):
    def __init__(self, **kwargs):
        AbstractDataReader.__init__(self, **kwargs)
        self._kwargs = kwargs
        self._reader = ODPSReader(
            project=self._kwargs["project"],
            access_id=self._kwargs["access_id"],
            access_key=self._kwargs["access_key"],
            endpoint=self._kwargs["endpoint"],
            table=self._kwargs["table"],
            num_processes=self._kwargs["num_processes"],
        )

    def read_records(self, task):
        records = self._reader.read_batch(
            start=task.start, end=task.end, columns=None
        )
        for batch in records:
            yield batch

    def create_shards(self):
        table_size = self._reader.get_table_size()
        records_per_task = self._kwargs["records_per_task"]
        shards = {}
        num_shards = math.floor(table_size / records_per_task)
        start_ind = 0
        for shard_id in range(num_shards):
            shards["shard_" + str(shard_id)] = (
                start_ind,
                start_ind + records_per_task - 1,
            )
            start_ind += records_per_task
        num_records_remain = table_size % records_per_task
        if num_records_remain != 0:
            shards["shard_" + str(num_shards)] = (
                start_ind,
                start_ind + num_records_remain,
            )
        return shards
