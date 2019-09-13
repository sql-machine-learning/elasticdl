from abc import ABC, abstractmethod
from contextlib import closing
import os

import recordio

from elasticdl.python.common.constants import Mode


class AbstractDataReader(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def read_records(self, task):
        """This method will be used in `TaskDataService` to read the records based on
        the information provided for a given task into a Python generator/iterator.

        Arguments:
            task: The current `Task` object that provides information on where
                to read the data for this task.
        """
        pass

    @abstractmethod
    def create_shards(self, mode):
        """This method creates the dictionary of shards where the keys are the
        shard names and the values are the number of records.

        Arguments:
            mode: The mode that indicates where the created shards will be used.
        """
        pass


class RecordIODataReader(AbstractDataReader):
    def __init__(self, **kwargs):
        AbstractDataReader.__init__(self, **kwargs)
        self._kwargs = kwargs

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

    def create_shards(self, mode):
        if mode == Mode.TRAINING:
            return self._collect_file_records_from_dir(self._kwargs["training_data_dir"])
        if mode == Mode.EVALUATION:
            return self._collect_file_records_from_dir(self._kwargs["evaluation_data_dir"])
        if mode == Mode.PREDICTION:
            return self._collect_file_records_from_dir(self._kwargs["prediction_data_dir"])

    @staticmethod
    def _collect_file_records_from_dir(data_dir):
        if not data_dir:
            return {}
        f_records = {}
        for f in os.listdir(data_dir):
            p = os.path.join(data_dir, f)
            with closing(recordio.Index(p)) as rio:
                f_records[p] = rio.num_records()
        return f_records
