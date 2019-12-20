import csv

import numpy as np
import tensorflow as tf

from elasticdl.python.data.data_reader import AbstractDataReader, Metadata


class CSVDataReader(AbstractDataReader):
    """This reader is used to read data from a csv file. It is convenient for
    user to locally run and debug a Keras model by using this reader.
    However, it can not be used with distribution strategy because it can not
    read data by line indices.
    """

    def __init__(self, **kwargs):
        AbstractDataReader.__init__(self, **kwargs)
        self._kwargs = kwargs

    def read_records(self, task):
        seq = self._kwargs.get("seq", ",")
        selected_columns = self._kwargs.get("columns")
        with open(task.shard_name, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=seq)
            csv_columns = next(csv_reader)
            selected_columns = (
                csv_columns if selected_columns is None else selected_columns
            )
            if not set(csv_columns) >= set(selected_columns):
                raise ValueError(
                    "The first line in the csv file must be column names and"
                    "the selected columns is not in the file. The selected"
                    "columns are {} and the columns in {} are {}".format(
                        selected_columns, task.shard_name, csv_columns
                    )
                )
            column_indices = _get_elements_indices(
                selected_columns, csv_columns
            )
            for line in csv_reader:
                line_elements = np.array(line, dtype=np.str)
                records = line_elements[column_indices].tolist()
                yield records

    def create_shards(self):
        pass

    @property
    def records_output_types(self):
        return tf.string

    @property
    def metadata(self):
        return Metadata(column_names=self._kwargs.get("columns"))


def _get_elements_indices(elements, element_set):
    indices = []
    for element in elements:
        indices.append(element_set.index(element))
    return indices
