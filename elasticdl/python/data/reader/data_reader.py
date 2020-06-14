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

from abc import ABC, abstractmethod

from elasticdl.python.common.dtypes import MAXCOMPUTE_DTYPE_TO_TF_DTYPE


class Metadata(object):
    """ Metadata of a dataset containing column name and dtype

    Attributes:
        column_names: A list with column names
        column_dtypes: A dict where the key is a column name
            and the value is a dtype. The dtypes are MaxCompute dtypes for a
            MaxCompute table or numpy dtypes for a CSV file.
    """

    def __init__(self, column_names, column_dtypes=None):
        self.column_names = column_names
        self.column_dtypes = column_dtypes

    @property
    def column_dtypes(self):
        return self.__column_dtypes

    @column_dtypes.setter
    def column_dtypes(self, column_dtypes):
        self.__column_dtypes = column_dtypes

    def get_tf_dtype_from_maxcompute_column(self, column_name):
        """Get TensorFlow dtype according to the column name in
        a MaxCompute table

        Args:
            column_name: The column name in a MaxCompute table.

        Returns:
            TensorFlow dtype
        """
        if self.column_dtypes is None:
            raise ValueError("The column dtypes has not been configured")

        maxcompute_dtype = self.column_dtypes.get(column_name, None)

        if maxcompute_dtype not in MAXCOMPUTE_DTYPE_TO_TF_DTYPE:
            raise ValueError(
                "Not support {} and only support {}".format(
                    maxcompute_dtype, list(MAXCOMPUTE_DTYPE_TO_TF_DTYPE.keys())
                )
            )
        return MAXCOMPUTE_DTYPE_TO_TF_DTYPE[maxcompute_dtype]


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


def check_required_kwargs(required_args, kwargs):
    missing_args = [k for k in required_args if k not in kwargs]
    if missing_args:
        raise ValueError(
            "The following required arguments are missing: %s"
            % ", ".join(missing_args)
        )
