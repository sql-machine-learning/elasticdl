## Design for Custom Data Source Support

This document describes the design for supporting custom data source in ElasticDL.

### Example Reader Class - `ODPSReader`

Reading data from ODPS data source is through the existing `ODPSReader` which has the following interface:

````python
class ODPSReader(object):
    def __init__(
        self,
        project,
        access_id,
        access_key,
        endpoint,
        table,
        partition=None,
        num_processes=None,
        options=None,
    ):
        """
        Constructs a `ODPSReader` instance.

        Args:
            project: Name of the ODPS project.
            access_id: ODPS user access ID.
            access_key: ODPS user access key.
            endpoint: ODPS cluster endpoint.
            table: ODPS table name.
            partition: ODPS table's partition.
            options: Other options passed to ODPS context.
            num_processes: Number of parallel processes on this worker.
                If `None`, use the number of cores.
        """
        pass

    def to_iterator(
        self,
        num_workers,
        worker_index,
        batch_size,
        epochs=1,
        shuffle=False,
        columns=None,
        cache_batch_count=None,
        limit=-1,
    ):
        """
        Load slices of ODPS table (partition of table if `partition`
        was specified) data with Python Generator.

        Args:
            num_workers: Total number of worker in the cluster.
            worker_index: Current index of the worker in the cluster.
            batch_size: Size of a slice.
            epochs: Repeat the data for this many times.
            shuffle: Whether to shuffle the data or rows.
            columns: The list of columns to load. If `None`,
                use all schema names of ODPS table.
            cache_batch_count: The cache batch count.
            limit: The limit for the table size to load.
        """
````

For example, if we have 5 workers in total, in the first worker, we can run the following
to load the ODPS table into a Python iterator where each batch contains 100 rows:

```python
reader = ODPSReader(...)
data_iterator = reader.to_iterator(num_workers=5, worker_index=0, batch_size=100)
for batch in data_iterator:
    print("Batch size %d\n. Data: %s" % (len(batch), batch))
```

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

* Rename ``shard_file_name`` in `Task` to `shard_name` so a task is independent with file names and `shard_name` will
serve as an identifier for the shard.
* Implement a method to create and dispatch tasks based on the provided data reader implementation.
* Implement an abstract custom data reader that accepts an custom data reader instance

A custom data reader implementation would look like the following:

```python
class CustomDataReader(object):
    def __init__(self, table_):
        self.reader = reader

    def generate_records(self, task):
        """This method will be used in `TaskDataService` to read the records for each task."""
        while True:
            record = self.reader.read(source=task.source,start=task.start, offset=task.end)
            if record:
                yield record
            else:
                break
```
* Should this be combined with dataset_fn?