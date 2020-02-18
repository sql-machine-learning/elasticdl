import tensorflow as tf

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

    def read_records(self, task):
        reader = self._get_reader(
            table_name=self._get_odps_table_name(task.shard_name)
        )
        if self._metadata.column_names is None:
            columns = self._kwargs.get("columns")
            self._metadata.column_names = (
                reader._odps_table.schema.names if columns is None else columns
            )

        for record in reader.record_generator_with_retry(
            start=task.start, end=task.end, columns=self._metadata.column_names
        ):
            yield record

    def create_shards(self):
        check_required_kwargs(["table", "records_per_task"], self._kwargs)
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
        check_required_kwargs(
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
