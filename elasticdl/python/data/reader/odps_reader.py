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

import tensorflow as tf
from odps import ODPS

from elasticdl.python.common.constants import Mode
from elasticdl.python.data.odps_io import ODPSReader
from elasticdl.python.data.reader.data_reader import (
    AbstractDataReader,
    Metadata,
    check_required_kwargs,
)


class ODPSDataReader(AbstractDataReader):
    def __init__(self, **kwargs):
        AbstractDataReader.__init__(self, **kwargs)
        self._kwargs = kwargs
        self._metadata = Metadata(column_names=None)
        self._table = self._kwargs["table"]
        self._columns = self._kwargs.get("columns")
        self._init_metadata()
        # Initialize an ODPS IO reader for each table with task type
        self._table_readers = dict()

    def _init_metadata(self):
        table_schema = self._get_table_schema()
        if self._metadata.column_names is None:
            self._metadata.column_names = (
                table_schema.names if self._columns is None else self._columns
            )

        if self._metadata.column_names:
            column_dtypes = {
                column_name: table_schema[column_name].type
                for column_name in self._metadata.column_names
            }
            self.metadata.column_dtypes = column_dtypes

    def read_records(self, task):
        task_table_name = self._get_odps_table_name(task.shard_name)
        self._init_reader(task_table_name, task.type)

        reader = self._table_readers[task_table_name][task.type]
        for record in reader.record_generator_with_retry(
            start=task.start, end=task.end, columns=self._metadata.column_names
        ):
            yield record

    def create_shards(self):
        check_required_kwargs(["table", "records_per_task"], self._kwargs)
        reader = self.get_odps_reader(self._kwargs["table"])
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

    def _init_reader(self, table_name, task_type):
        if (
            table_name in self._table_readers
            and task_type in self._table_readers[table_name]
        ):
            return

        self._table_readers.setdefault(table_name, {})

        check_required_kwargs(
            ["project", "access_id", "access_key"], self._kwargs
        )
        reader = self.get_odps_reader(table_name)

        # There may be weird errors if tasks with the same table
        # and different type use the same reader.
        self._table_readers[table_name][task_type] = reader

    def get_odps_reader(self, table_name):
        return ODPSReader(
            project=self._kwargs["project"],
            access_id=self._kwargs["access_id"],
            access_key=self._kwargs["access_key"],
            table=table_name,
            endpoint=self._kwargs.get("endpoint"),
            partition=self._kwargs.get("partition", None),
            num_processes=self._kwargs.get("num_processes", 1),
            options={
                "odps.options.tunnel.endpoint": self._kwargs.get(
                    "tunnel_endpoint", None
                )
            },
        )

    def _get_table_schema(self):
        odps_client = ODPS(
            access_id=self._kwargs["access_id"],
            secret_access_key=self._kwargs["access_key"],
            project=self._kwargs["project"],
            endpoint=self._kwargs.get("endpoint"),
        )
        odps_table = odps_client.get_table(self._kwargs["table"])
        return odps_table.schema

    @staticmethod
    def _get_odps_table_name(shard_name):
        return shard_name.split(":")[0]

    def default_dataset_fn(self):
        check_required_kwargs(["label_col"], self._kwargs)

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


class ParallelODPSDataReader(ODPSDataReader):
    """Use multi-process to download records from a MaxCompute table
    """

    def __init__(self, parse_fn, **kwargs):
        ODPSDataReader.__init__(self, **kwargs)
        self.py_parse_data = parse_fn

    def parallel_record_records(
        self, task, num_processes, shard_size, transform_fn
    ):
        check_required_kwargs(
            ["project", "access_id", "access_key"], self._kwargs
        )
        start = task.start
        end = task.end
        table = self._get_odps_table_name(task.shard_name)
        table = table.split(".")[1]
        project = self._kwargs["project"]
        access_id = self._kwargs["access_id"]
        access_key = self._kwargs["access_key"]
        endpoint = self._kwargs.get("endpoint")
        partition = self._kwargs.get("partition", None)
        columns = self._kwargs.get("columns", None)
        pd = ODPSReader(
            access_id=access_id,
            access_key=access_key,
            project=project,
            endpoint=endpoint,
            table=table,
            partition=partition,
            num_processes=num_processes,
            transform_fn=transform_fn,
            columns=columns,
        )
        pd.reset((start, end - start), shard_size)
        shard_count = pd.get_shards_count()
        for i in range(shard_count):
            records = pd.get_records()
            for record in records:
                yield record
        pd.stop()

    def read_records(self, task):
        shard_size = (task.end - task.start) // 4
        record_gen = self.parallel_record_records(
            task=task,
            num_processes=4,
            shard_size=shard_size,
            transform_fn=self.py_parse_data,
        )
        for record in record_gen:
            yield record

    @property
    def records_output_types(self):
        return tf.string
