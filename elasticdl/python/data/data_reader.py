from abc import ABC, abstractmethod


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


def check_required_kwargs(required_args, kwargs):
    missing_args = [k for k in required_args if k not in kwargs]
    if missing_args:
        raise ValueError(
            "The following required arguments are missing: %s"
            % ", ".join(missing_args)
        )
