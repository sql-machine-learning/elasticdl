## Design for Custom Data Source Support

This document describes the design for supporting custom data source in ElasticDL.

[RecordIO](https://github.com/wangkuiyi/recordio) is a file format that supports dynamic sharding for
performing fault-tolerant distributed computing or elastic scheduling of distributed computing jobs. It is
currently the only supported data format for ElasticDL.

However, there may be a lot of I/O overhead to convert from existing data sources to RecordIO format and requires additional storage
for the converted RecordIO files. In addition, saving the converted RecordIO files to disk will also lead to potential security problems
since the file access must be controlled properly if the Kubernetes cluster is shared across different teams and business units.
Some distributed file system implementations may not support access control yet.

For example, in order to fetch data from [ODPS](https://www.alibabacloud.com/product/maxcompute), we would first
need to read the data from ODPS table and then convert it to RecordIO format, which could waste a lot of time.
Instead, we could expose necessary pieces in ElasticDL to users so that they can implement their own data reading logic to
avoid having to write RecordIO files to disk. This way ElasticDL can perform tasks while reading data into memory asynchronously.

### Support Custom Data Source in ElasticDL

There are a couple of places in the the current codebase (as of commit [77cc87a](https://github.com/sql-machine-learning/elasticdl/tree/77cc87a90eec54db565849f0ae07d271fd957190))
that are coupled heavily with RecordIO, for example:

* When we create and dispatch tasks via ``_make_task_dispatcher()``, we rely on scanning directories that contain
RecordIO files and reading the index for each file to obtain dictionaries of `{file_path: num_records}`.
We then use this information to create tasks.
* ``Worker`` starts a `TaskDataService` that generates a `tf.data.Dataset` instance that contains the actual data.
The data is fetched in `TaskDataService._gen()` by opening a RecordIO reader to read the data offsets that each task
we previously created.

In order to support custom data sources in ElasticDL, we propose to make the following changes:

* Rename ``shard_file_name`` in `Task` protobuf definition to `shard_name` so a task is independent with file names and `shard_name` will
serve as an identifier for the shard.
* Implement an abstract interface named `AbstractDataReader` that users can implement in their model definition file.
The interface would look like the following:

```python
class AbstractDataReader(object):
    def __init__(self, *kwargs):
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
```

Users can then implement a custom data reader implementation similar to the following:

```python
class CustomDataReader(AbstractDataReader):
    def __init__(self, *kwargs):
        self.reader = ...

    def read_records(self, task):
        while True:
            record = self.reader.read(source=task.shard_name, start=task.start, offset=task.end)
            if record:
                yield record
            else:
                break

    def create_shards(self, mode):
        if mode == Mode.TRAINING:
            return training_shards
        elif mode == Mode.EVALUATION:
            return evaluation_shards
        else:
            return prediction_shards
```
* Implement a `RecordIODataReader` that can be used to read RecordIO files, serves as the default
data reader for ElasticDL, and preserves the current ElasticDL functionality.
* Implement a `ODPSDataReader` that implements the abstract methods in `AbstractDataReader` and
reuses the existing `ODPSReader` we currently have.
* Keep the existing CLI arguments `--training_data_dir`, `--evaluation_data_dir`, and `--prediction_data_dir` that
will be used for specifying paths to RecordIO files if no custom data reader is specified.
* Add CLI argument `--custom_data_reader_params` to pass parameters to the constructor of user-defined `CustomDataReader`
similar to the existing `--model_params` used for passing parameters to model constructor.
