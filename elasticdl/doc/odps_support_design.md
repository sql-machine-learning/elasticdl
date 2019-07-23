## Design for ODPS Data Source Support

This document describes the design for supporting ODPS data source in ElasticDL.

### Existing `ODPSReader` Class

The interface to read data from ODPS with the existing `ODPSReader` is defined as follows:

````python
class ODPSReader(object):
    def __init__(self, project, access_id, access_key, endpoint,
                 table, partition=None, options=None):
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
        """
        pass

    def to_iterator(self, num_workers, worker_index, batch_size, epochs=1, shuffle=False,
                    columns=None, table_size_limit=-1, num_processes=None):
        """
        Load slices of ODPS table (partition of table if `partition` was specified) data with Python iterator.

        Args:
            num_workers: Total number of worker in the cluster.
            worker_index: Current index of the worker in the cluster.
            batch_size: Size of a slice.
            epochs: Repeat the data for this many times.
            shuffle: Whether to shuffle the data or rows.
            columns: The list of columns to load. If `None`, use all schema names of ODPS table.
            table_size_limit: The limit for the table size to load.
            num_processes: Number of parallel processes on this worker. If `None`, use the number of cores.
        """
        pass
````

For example, if we have 5 workers in total, in the first worker, we can run the following
to load the ODPS table into a Python iterator where each batch contains 100 rows:

```python
reader = ODPSReader(...)
data_iterator = reader.to_iterator(num_workers=5, worker_index=0, batch_size=100)
for batch in data_iterator:
    print("Batch size %d\n. Data: %s" % (len(batch), batch))
```

### Support ODPS Data Source in ElasticDL

The current `Worker` relies heavily on `TaskDispatcher` and RecordIO which overlaps with the
existing `ODPSReader` so we could not use the above existing `ODPSReader` directly. For example,
`Worker` does the following related steps:

1. `recordio.Scanner(task.shard_file_name, task.start, task.end - task.start)`
to create the reader for RecordIO data source.
2. `self._get_batch(reader, task.minibatch_size)` to
get one batch based on the `batch_size` that users specified.
3. Use the user-provided `input_fn` to
fetch the features and labels from this batch.

Here's a list of things we need to do in order to support ODPS data source:

1. Create `ODPSReader` based on user-provided ODPS information such as ODPS project name and credentials.
2. Implement a method to create training and evaluation shards based on table name and column names instead of
RecordIO data directories, and then pass the shards to `TaskDispatcher`.
3. Modify `Worker` to support instantiating a `ODPSReader` in addition to RecordIO reader.
4. Implement `ODPSReader.record()` for `Worker._get_batch()` to use. Alternatively, we can also re-implement
`Worker._get_batch()` so it can get the whole batch of data rows if the data source is ODPS. This is because
the current implementation of `Worker._get_batch()` method contains a for loop that reads one record at a time
which is inefficient.

Once the above work is done and we have a clearer picture, we could then think about how to allow users to plug
in their custom data readers like `ODPSReader` so that they don't have to convert data to RecordIO format, which
could avoid the IO overhead. This should be discussed further in high-level API designs.
