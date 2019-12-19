import os
from abc import ABC, abstractmethod
from contextlib import closing

import recordio
import tensorflow as tf

from elasticdl.python.common.constants import Mode, ODPSConfig
from elasticdl.python.data.odps_io import ODPSReader, is_odps_configured


class Metadata(object):
    def __init__(self, column_names):
        self.column_names = column_names


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

    @property
    def records_output_types(self):
        """This method returns the output data types used for
        `tf.data.Dataset.from_generator` when creating the
        `tf.data.Dataset` object from the generator created
        by `read_records()`. Note that the returned output types
        should be a nested structure of `tf.DType` objects corresponding
        to each component of an element yielded by the created generator.
        """
        return None

    @property
    def metadata(self):
        """This method returns the `Metadata` object that contains
         some metadata collected for the read records, such as the
         list of column names."""
        return Metadata(column_names=None)


class RecordIODataReader(AbstractDataReader):
    def __init__(self, **kwargs):
        AbstractDataReader.__init__(self, **kwargs)
        self._kwargs = kwargs
        _check_required_kwargs(["data_dir"], self._kwargs)

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
        f_records = {}
        for f in os.listdir(data_dir):
            p = os.path.join(data_dir, f)
            with closing(recordio.Index(p)) as rio:
                f_records[p] = (start_ind, rio.num_records())
        return f_records

    @property
    def records_output_types(self):
        return tf.string

    @property
    def metadata(self):
        return Metadata(column_names=None)


class ODPSDataReader(AbstractDataReader):
    def __init__(self, **kwargs):
        AbstractDataReader.__init__(self, **kwargs)
        self._kwargs = kwargs
        self._metadata = Metadata(column_names=None)

    def read_records(self, task):
        reader = self._get_reader(
            table_name=self._get_odps_table_name(task.shard_name)
        )
        if self._metadata.column_names is None:
            columns = self._kwargs.get("columns")
            self._metadata.column_names = (
                reader._odps_table.schema.names if columns is None else columns
            )
        records = reader.read_batch(
            start=task.start, end=task.end, columns=self._metadata.column_names
        )
        for batch in records:
            yield batch

    def create_shards(self):
        _check_required_kwargs(["table", "records_per_task"], self._kwargs)
        reader = self._get_reader(self._kwargs["table"])
        shard_name_prefix = self._kwargs["table"] + ":shard_"
        table_size = reader.get_table_size()
        records_per_task = self._kwargs["records_per_task"]
        shards = {}
        num_shards = table_size // records_per_task
        start_ind = 0
        for shard_id in range(num_shards):
            shards[shard_name_prefix + str(shard_id)] = (
                start_ind,
                records_per_task,
            )
            start_ind += records_per_task
        num_records_left = table_size % records_per_task
        if num_records_left != 0:
            shards[shard_name_prefix + str(num_shards)] = (
                start_ind,
                num_records_left,
            )
        return shards

    @property
    def records_output_types(self):
        return tf.string

    @property
    def metadata(self):
        return self._metadata

    def _get_reader(self, table_name):
        _check_required_kwargs(
            ["project", "access_id", "access_key"], self._kwargs
        )
        return ODPSReader(
            project=self._kwargs["project"],
            access_id=self._kwargs["access_id"],
            access_key=self._kwargs["access_key"],
            table=table_name,
            endpoint=self._kwargs.get("endpoint"),
            partition=self._kwargs.get("partition", None),
            num_processes=self._kwargs.get("num_processes", 1),
        )

    @staticmethod
    def _get_odps_table_name(shard_name):
        return shard_name.split(":")[0]

    def default_dataset_fn(self):
        _check_required_kwargs(["label_col"], self._kwargs)

        def dataset_fn(dataset, mode, metadata):
            def _parse_data(record):
                label_col_name = self._kwargs["label_col"]
                record = tf.strings.to_number(record, tf.float32)

                def _get_features_without_labels(
                    record, label_col_idx, features_shape
                ):
                    features = [
                        record[:label_col_idx],
                        record[label_col_idx + 1 :],  # noqa: E203
                    ]
                    features = tf.concat(features, -1)
                    return tf.reshape(features, features_shape)

                features_shape = (len(metadata.column_names) - 1, 1)
                labels_shape = (1,)
                if mode == Mode.PREDICTION:
                    if label_col_name in metadata.column_names:
                        label_col_idx = metadata.column_names.index(
                            label_col_name
                        )
                        return _get_features_without_labels(
                            record, label_col_idx, features_shape
                        )
                    else:
                        return tf.reshape(record, features_shape)
                else:
                    if label_col_name not in metadata.column_names:
                        raise ValueError(
                            "Missing the label column '%s' in the retrieved "
                            "ODPS table during %s mode."
                            % (label_col_name, mode)
                        )
                    label_col_idx = metadata.column_names.index(label_col_name)
                    labels = tf.reshape(record[label_col_idx], labels_shape)
                    return (
                        _get_features_without_labels(
                            record, label_col_idx, features_shape
                        ),
                        labels,
                    )

            dataset = dataset.map(_parse_data)

            if mode == Mode.TRAINING:
                dataset = dataset.shuffle(buffer_size=200)
            return dataset

        return dataset_fn


def create_data_reader(data_origin, records_per_task=None, **kwargs):
    if is_odps_configured():
        return ODPSDataReader(
            project=os.environ[ODPSConfig.PROJECT_NAME],
            access_id=os.environ[ODPSConfig.ACCESS_ID],
            access_key=os.environ[ODPSConfig.ACCESS_KEY],
            table=data_origin,
            endpoint=os.environ.get(ODPSConfig.ENDPOINT),
            records_per_task=records_per_task,
            **kwargs,
        )
    else:
        return RecordIODataReader(data_dir=data_origin)


def _check_required_kwargs(required_args, kwargs):
    missing_args = [k for k in required_args if k not in kwargs]
    if missing_args:
        raise ValueError(
            "The following required arguments are missing: %s"
            % ", ".join(missing_args)
        )
