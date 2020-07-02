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

import numpy as np
import tensorflow as tf

from elasticdl.python.data.reader.data_reader import (
    AbstractDataReader,
    Metadata,
    check_required_kwargs,
)


class CSVDataReader(AbstractDataReader):
    """This reader is used to read data from a csv file. It is convenient for
    user to locally run and debug a Keras model by using this reader.
    However, it cannot be used with distribution strategy because it cannot
    read data by line indices.
    """

    def __init__(self, **kwargs):
        """
        Args:
            kwargs should contains "sep" and "columns" like
            'sep=",",column=["sepal.length", "sepal.width", "variety"]'
        """
        AbstractDataReader.__init__(self, **kwargs)
        check_required_kwargs(["sep", "columns"], kwargs)
        self.sep = kwargs.get("sep", ",")
        self.selected_columns = kwargs.get("columns", None)

    def read_records(self, task):
        with open(task.shard_name, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=self.sep)
            csv_columns = next(csv_reader)
            selected_columns = (
                csv_columns
                if self.selected_columns is None
                else self.selected_columns
            )
            if not set(selected_columns).issubset(set(csv_columns)):
                raise ValueError(
                    "The first line in the csv file must be column names and "
                    "the selected columns are not in the file. The selected "
                    "columns are {} and the columns in {} are {}".format(
                        selected_columns, task.shard_name, csv_columns
                    )
                )
            column_indices = [csv_columns.index(e) for e in selected_columns]
            for line in csv_reader:
                line_elements = np.array(line, dtype=np.str)
                yield line_elements[column_indices].tolist()

    def create_shards(self):
        pass

    @property
    def records_output_types(self):
        return tf.string

    @property
    def metadata(self):
        return Metadata(column_names=self.selected_columns)
